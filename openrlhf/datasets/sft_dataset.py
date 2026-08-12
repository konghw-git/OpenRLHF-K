import logging
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences
from openrlhf.utils.vlm_utils import MM_SKIP_KEYS, _is_base64_image, load_images

logger = logging.getLogger(__name__)

IMAGE_PAD_TOKEN = "<|image_pad|>"


def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiturn=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # For VLMs, get_tokenizer returns an AutoProcessor whose __call__ takes `images` as its
        # first positional argument. Every text-only tokenization below must therefore go through
        # the inner tokenizer, or the text is parsed as an image reference. For text-only models
        # processor is None and text_tokenizer is tokenizer, so behaviour is unchanged.
        self.processor = tokenizer if hasattr(tokenizer, "image_processor") else None
        self.text_tokenizer = tokenizer.tokenizer if self.processor is not None else tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn
        self._truncation_warned = False
        self._caliber_fallback_warned = False
        self._image_pad_id = None

        if self.processor is not None:
            self._apply_image_pixel_limits()
            pad_id = self.text_tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
            if pad_id != self.text_tokenizer.unk_token_id:
                self._image_pad_id = pad_id

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.output_key = getattr(self.strategy.args.data, "output_key", None)
        self.image_key = getattr(self.strategy.args.data, "image_key", "images")
        self.apply_chat_template = getattr(self.strategy.args.data, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args.data, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        num_raw = len(processed_dataset)
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
        num_dropped = num_raw - len(processed_dataset)
        if num_dropped:
            # Make the length filter visible: dropping is correct, doing it silently is not.
            self.strategy.print(
                f"SFTDataset: dropped {num_dropped}/{num_raw} samples "
                f"(empty, or prompt+response longer than max_len={self.max_length})"
            )

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.images = processed_dataset["images"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None

        self.has_images = self.processor is not None and any(self.images)
        self._check_unsupported_combinations()

    def _check_unsupported_combinations(self):
        """Reject option combinations that images are known to break.

        Checked here rather than at argument-parse time: unlike the RL path there is no VLM
        flag on the SFT command line, so whether images are involved is only known once the
        dataset is built. Both the train and eval datasets go through this.
        """
        if not self.has_images:
            return

        if self.multiturn:
            # response_ranges are computed with the TEXT tokenizer, so they do not account for
            # the image placeholder expansion -> every range would be shifted left. Refuse
            # rather than train on a silently misaligned loss mask.
            raise NotImplementedError(
                "Multiturn SFT with images is not supported yet (response_ranges are text-caliber "
                "and would misalign the loss mask by the image placeholder expansion)."
            )

        if getattr(getattr(self.strategy.args, "ds", None), "ring_attn_size", 1) > 1:
            # Without this the run starts fine and Actor.forward only asserts on the first
            # micro-batch that carries images, which on a mixed text/image corpus can be many
            # steps in.
            raise ValueError(
                "SFT with images does not support ring attention: sequence parallelism splits a "
                "single image's tokens across ranks, breaking image-token/pixel_values alignment. "
                "Set --ds.ring_attn_size 1."
            )

    def _apply_image_pixel_limits(self):
        """Override the image processor's resolution limits from --data.image_{max,min}_pixels.

        Uncapped images expand into very large placeholder runs, which overruns max_length or
        OOMs. Only applies to processors that express their limits as longest/shortest_edge
        (Qwen*-VL); processors sized by height/width (Siglip/CLIP) are left untouched.
        """
        max_pixels = getattr(self.strategy.args.data, "image_max_pixels", None)
        min_pixels = getattr(self.strategy.args.data, "image_min_pixels", None)
        if max_pixels is None and min_pixels is None:
            return
        size = self.processor.image_processor.size
        if getattr(size, "longest_edge", None) is None and (not isinstance(size, dict) or "longest_edge" not in size):
            self.strategy.print(
                "SFTDataset: --data.image_{max,min}_pixels ignored; this image processor is not "
                "sized by longest_edge/shortest_edge."
            )
            return
        # Merge rather than replace, so setting only one bound keeps the processor's other one.
        limits = {k: v for k, v in dict(size).items() if v is not None}
        if max_pixels is not None:
            limits["longest_edge"] = max_pixels
        if min_pixels is not None:
            limits["shortest_edge"] = min_pixels
        self.processor.image_processor.size = limits

    def _num_image_placeholder_tokens(self, images):
        """Number of tokens the image placeholders of one sample expand into, summed.

        Uses the processor's size->patch utility, so only the image *header* is read -- no
        decode, no resize. Returns None when it cannot be determined, in which case callers
        fall back to text caliber.

        An unreadable LOCAL PATH raises instead of falling back: falling back would keep the
        row, and it would then die in ``__getitem__`` with an ``IndexError`` from deep inside
        transformers, minutes into training and with no path in the message.
        """
        if self.processor is None or not images:
            return 0

        sizes = []
        for ref in images if isinstance(images, list) else [images]:
            if ref is None:
                continue
            if isinstance(ref, Image.Image):
                w, h = ref.size
            elif isinstance(ref, str) and not ref.startswith(("http://", "https://")) and not _is_base64_image(ref):
                try:
                    with Image.open(ref) as im:  # lazy: reads the header only
                        w, h = im.size
                except Exception as e:
                    raise FileNotFoundError(f"SFTDataset: unreadable image reference {ref!r} ({e})") from e
            else:
                # URL / base64 / raw bytes: no cheap header read (a URL would mean a network
                # fetch per row) -> text caliber. Say so, since a length filter that quietly
                # changes caliber reads as "all accounted for" when it is not.
                if not self._caliber_fallback_warned:
                    self._caliber_fallback_warned = True
                    logger.warning(
                        f"SFTDataset: image reference {type(ref).__name__} is not a local path "
                        "(URL/base64/bytes); image expansion is NOT accounted for in the length "
                        "filter for such rows -- they may be truncated at max_length."
                    )
                return None
            sizes.append([h, w])
        if not sizes:
            return 0
        try:
            mm_data = self.processor._get_num_multimodal_tokens(image_sizes=sizes)
            return int(sum(mm_data.num_image_tokens))
        except Exception as e:  # unknown processor API -> text caliber is an acceptable fallback
            logger.warning(f"Cannot pre-compute image token count ({e}); falling back to text-caliber length")
            return None

    def process_data(self, data):
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "You should put the whole trajectory into data[input_key] and do not set output_key"
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            response_ranges = []
            # The spans below are computed from per-turn prefix re-renderings, while the text
            # actually trained on is the one-shot rendering of the whole trajectory. A chat
            # template may render a turn differently in the two passes -- e.g. the
            # Qwen3-Thinking template drops empty <think> blocks from non-final turns and
            # strips reasoning from turns before the last user query -- which shifts every
            # later span and silently supervises the tool observations instead of the
            # assistant turns, with a normal-looking loss curve. Both prefix properties are
            # asserted per turn so a mismatched trajectory fails at load time, naming the turn.
            full_rendered = apply_chat_template(data[input_key], tokenize=False)
            for idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt = apply_chat_template(data[input_key][:idx], tokenize=False, add_generation_prompt=True)
                    upto = apply_chat_template(data[input_key][: idx + 1], tokenize=False)
                    if not (upto.startswith(prompt) and full_rendered.startswith(upto)):
                        raise ValueError(
                            f"SFTDataset multiturn: the chat template renders assistant turn {idx} "
                            "differently as a prefix than inside the full conversation, so the "
                            "loss-mask spans would be misaligned. Known triggers with thinking "
                            "templates: an assistant turn with an empty or missing <think> block, "
                            "or more than one real user question in one trajectory."
                        )
                    response = upto[len(prompt) :]

                    start_idx = (
                        self.text_tokenizer(
                            prompt,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )

                    end_idx = (
                        start_idx
                        + self.text_tokenizer(
                            response,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                        - 1
                    )
                    response_ranges.append((start_idx, end_idx))  # left close right close

        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        )

        images = data.get(self.image_key, None) if self.image_key else None
        if images is not None and not isinstance(images, list):
            images = [images]

        if not self.pretrain_mode:
            prompt_token = self.text_tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            drop = not prompt or not response or prompt_ids_len >= self.max_length - 2

            # Also filter on the TOTAL length. Checking only the prompt lets a short prompt with
            # a long (e.g. long-CoT) target through, where it is silently truncated at
            # max_length -- i.e. the model is trained on a derivation that stops mid-way.
            if not drop:
                response_ids_len = len(
                    self.text_tokenizer(response, padding=False, truncation=False, add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                # VLM: one <|image_pad|> in the prompt expands into grid/merge**2 tokens at
                # __getitem__ time, so the text caliber above underestimates image samples.
                # _image_pad_id is None when the placeholder is absent from the vocab, i.e. when
                # this accounting cannot be trusted; _expanded_prompt_ids_len uses the same guard.
                extra = 0
                if images and self.processor is not None:
                    n_expanded = self._num_image_placeholder_tokens(images)
                    if n_expanded is not None and self._image_pad_id is not None:
                        extra = n_expanded - prompt.count(IMAGE_PAD_TOKEN)
                # +2: the eos appended in __getitem__, plus one token of slack for the
                # prompt/response boundary re-tokenization.
                if prompt_ids_len + response_ids_len + extra + 2 > self.max_length:
                    drop = True
            if drop:
                prompt = None
        else:
            prompt_ids_len = 0

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "images": images,
            "response_ranges": response_ranges if self.multiturn else None,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.text_tokenizer.eos_token):
                text += " " + self.text_tokenizer.eos_token
        else:
            text = prompt

        tokenize_kwargs = dict(
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        images = self.images[idx] if (self.processor is not None and self.images is not None) else None
        if images:
            pil_images = load_images(images)  # same loading path as the RL side
            # load_images() only warns on a bad reference and drops it, and a short image list
            # does not degrade gracefully: zero images raises IndexError deep in transformers,
            # a partial list silently desyncs the placeholder count from the grids. Fail here
            # instead, naming the reference.
            n_expected = sum(1 for r in (images if isinstance(images, list) else [images]) if r is not None)
            if len(pil_images) != n_expected:
                raise ValueError(
                    f"SFTDataset: sample {idx} declares {n_expected} image(s) but only "
                    f"{len(pil_images)} loaded (see the load_images warning above); refs={images!r}"
                )
            enc = self.processor(images=pil_images, text=[text], **tokenize_kwargs)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            mm_inputs = {k: v for k, v in enc.items() if k not in MM_SKIP_KEYS}
            # self.prompt_ids_lens[idx] is text caliber and excludes the placeholder expansion:
            # reusing it here would shift the loss mask left by the grid size and supervise the
            # image pad tokens, with no error and a normal-looking loss curve. Recover the real
            # boundary from the grids we already have, rather than a second image_processor call.
            prompt_ids_len = self._expanded_prompt_ids_len(idx, mm_inputs.get("image_grid_thw"))
        else:
            input_token = self.text_tokenizer(text, **tokenize_kwargs)
            input_ids = input_token["input_ids"]
            attention_mask = input_token["attention_mask"]
            mm_inputs = {}
            prompt_ids_len = self.prompt_ids_lens[idx]

        loss_mask = self.get_loss_mask(input_ids, idx, prompt_ids_len)

        if not self.pretrain_mode:
            eos_token_id = self.text_tokenizer.eos_token_id
            # `truncation=True` may have cut the sample at max_length, and forcing EOS onto the
            # last token of a truncated sample teaches the model to stop mid-derivation. Only do
            # it when the sample actually ended -- where it is a no-op anyway, since `text` ends
            # with eos. Over-long samples should have been dropped by the filter in process_data.
            if int(input_ids[0][-1]) != eos_token_id:
                if input_ids.shape[-1] < self.max_length:
                    input_ids[0][-1] = eos_token_id
                    attention_mask[0][-1] = True
                elif not self._truncation_warned:
                    self._truncation_warned = True
                    logger.warning(
                        f"SFTDataset: sample {idx} was truncated at max_length={self.max_length} "
                        "(the length filter underestimated it, e.g. image placeholder expansion). "
                        "Leaving it without a forced EOS -- fix the data instead."
                    )
        return input_ids, attention_mask, loss_mask, mm_inputs

    def _expanded_prompt_ids_len(self, idx, image_grid_thw):
        """Processor-caliber prompt length for a multimodal sample (see __getitem__):

            prompt_len(processor) = prompt_len(text) - n_placeholders + sum(prod(grid)) / merge**2

        ``self.prompt_ids_lens[idx]`` is the untruncated text-caliber length: process_data
        tokenizes with truncation at max_length, and every row that survived the filter is
        shorter than that. Counting placeholders in the prompt *string* rather than
        re-tokenizing keeps one caliber for the same quantity and keeps a full prompt
        re-tokenization out of the dataloader hot path.
        """
        prompt_ids_len = self.prompt_ids_lens[idx]
        if image_grid_thw is None or self._image_pad_id is None:
            return prompt_ids_len
        merge = getattr(self.processor.image_processor, "merge_size", 2)
        grid_tokens = sum(int(g[0]) * int(g[1]) * int(g[2]) // (merge**2) for g in image_grid_thw)
        return prompt_ids_len - self.prompts[idx].count(IMAGE_PAD_TOKEN) + grid_tokens

    def get_loss_mask(self, input_ids, idx, prompt_ids_len=None):
        if self.pretrain_mode:
            return torch.ones_like(input_ids, dtype=torch.float32)  # shape:[1, seq_len]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if not self.multiturn:
            if prompt_ids_len is None:
                prompt_ids_len = self.prompt_ids_lens[idx]
            loss_mask[0, prompt_ids_len - 1 : -1] = 1
        else:
            response_ranges = self.response_ranges[idx]
            seq_len = input_ids.shape[1]
            for start_idx, end_idx in response_ranges:
                # Same calibre as the single-turn branch above: the position that would
                # predict past the end of the sequence (a PAD after collation) carries no
                # loss. The last turn's end_idx points one past it because __getitem__
                # rstrips the trailing newline of the final rendering.
                loss_mask[0, start_idx - 1 : min(end_idx, seq_len - 1)] = 1
        return loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []
        mm_list = []

        for input_id, attention_mask, loss_mask, mm_inputs in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)
            mm_list.append(mm_inputs)

        input_ids = zero_pad_sequences(input_ids, "right", self.text_tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")

        # Concatenate only the rows that actually carry multimodal tensors, along the batch dim.
        # The model re-associates them with their rows via the <|image_pad|> runs in input_ids,
        # so a mixed text/image batch works. All-text batch -> {}, never a None value: that
        # would be forwarded as a kwarg and crash the model.
        mm_inputs = {}
        for mm in mm_list:
            for k, v in mm.items():
                mm_inputs.setdefault(k, []).append(v)
        mm_inputs = {k: torch.cat(v, dim=0) for k, v in mm_inputs.items() if v}
        return input_ids, attention_masks, loss_masks, mm_inputs
