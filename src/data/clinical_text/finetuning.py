"""
Clinical text fine-tuning via PEFT/LoRA.

Wraps `peft.LoraConfig` and HuggingFace `Trainer` to fine-tune biomedical
language models on downstream tasks (NER, classification, causal LM)
without reimplementing training loop logic.
"""
from __future__ import annotations

from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

from opentbtk.core.base import BaseFineTuner
from opentbtk.core.errors import ProcessingError
from opentbtk.core.registry import FINETUNER_REGISTRY

log = structlog.get_logger(__name__)

TaskType = Literal["token_classification", "sequence_classification", "causal_lm"]


class ClinicalFineTuningConfig(BaseModel):
    """Configuration for clinical text LoRA fine-tuning."""

    model_config = {"frozen": True}

    base_model: str = Field(
        ..., description="HuggingFace model ID or local path, e.g. 'emilyalsentzer/Bio_ClinicalBERT'."
    )
    task: TaskType = Field(..., description="Fine-tuning task type.")
    num_labels: int | None = Field(
        None, description="Required for token_classification/sequence_classification."
    )
    lora_rank: int = Field(8, ge=1, description="LoRA rank (r).")
    lora_alpha: int = Field(16, ge=1, description="LoRA alpha scaling factor.")
    lora_dropout: float = Field(0.1, ge=0.0, le=1.0)
    target_modules: list[str] | None = Field(
        None,
        description="Module names to apply LoRA to. If None, uses task-appropriate defaults.",
    )
    learning_rate: float = Field(2e-4, gt=0)
    num_epochs: int = Field(3, ge=1)
    per_device_batch_size: int = Field(8, ge=1)
    fp16: bool = Field(False, description="Use mixed-precision training (requires CUDA).")
    output_dir: str = Field("./opentbtk_finetune_output")


_DEFAULT_TARGET_MODULES: dict[str, list[str]] = {
    "token_classification": ["query", "value"],
    "sequence_classification": ["query", "value"],
    "causal_lm": ["q_proj", "v_proj"],
}


@FINETUNER_REGISTRY.register("finetuner.clinical_text.lora")
class ClinicalTextFineTuner(BaseFineTuner):
    """LoRA fine-tuning adapter for biomedical text models.

    Supports three task types:
    - `token_classification`: NER, e.g. fine-tuning ClinicalBERT for
      clinical entity recognition.
    - `sequence_classification`: document/sentence classification, e.g.
      note type classification, risk stratification.
    - `causal_lm`: generative fine-tuning of decoder models (e.g., BioGPT)
      for clinical text generation/summarization.

    This adapter wraps `peft` + `transformers.Trainer` — it does not
    reimplement training logic. Use `prepare()` to load and wrap the base
    model, then `train()` to run fine-tuning, then `save()` to persist
    adapter weights (not the full base model — LoRA adapters are small).
    """

    def prepare(self, model_name_or_path: str, **kwargs: Any) -> Any:
        """Load a base model and wrap it with a LoRA adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            **kwargs: Must include `task` ("token_classification",
                "sequence_classification", or "causal_lm"). May include
                `num_labels`, `lora_rank`, `lora_alpha`, `lora_dropout`,
                `target_modules`.

        Returns:
            A PEFT-wrapped model ready for training.

        Raises:
            ProcessingError: If required dependencies are missing or task
                is invalid.
        """
        try:
            from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model
            from transformers import (
                AutoModelForTokenClassification,
                AutoModelForSequenceClassification,
                AutoModelForCausalLM,
            )
        except ImportError as e:
            raise ProcessingError(
                "peft and transformers are required for fine-tuning. "
                "Install with: pip install opentbtk[clinical_text]",
            ) from e

        task: TaskType = kwargs.get("task", "sequence_classification")
        num_labels = kwargs.get("num_labels")
        lora_rank = kwargs.get("lora_rank", 8)
        lora_alpha = kwargs.get("lora_alpha", 16)
        lora_dropout = kwargs.get("lora_dropout", 0.1)
        target_modules = kwargs.get("target_modules") or _DEFAULT_TARGET_MODULES[task]

        if task in ("token_classification", "sequence_classification") and not num_labels:
            raise ProcessingError(
                f"task='{task}' requires 'num_labels' to be specified.",
            )

        try:
            if task == "token_classification":
                base_model = AutoModelForTokenClassification.from_pretrained(
                    model_name_or_path, num_labels=num_labels
                )
                peft_task = PeftTaskType.TOKEN_CLS
            elif task == "sequence_classification":
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path, num_labels=num_labels
                )
                peft_task = PeftTaskType.SEQ_CLS
            elif task == "causal_lm":
                base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
                peft_task = PeftTaskType.CAUSAL_LM
            else:
                raise ProcessingError(f"Unknown task type: {task}")
        except Exception as e:
            raise ProcessingError(
                f"Failed to load base model '{model_name_or_path}' for task '{task}'",
                context={"model_name_or_path": model_name_or_path, "task": task},
            ) from e

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=peft_task,
        )
        peft_model = get_peft_model(base_model, lora_config)

        log.info(
            "finetuner.prepared",
            model=model_name_or_path,
            task=task,
            lora_rank=lora_rank,
            target_modules=target_modules,
        )
        return peft_model

    def train(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run LoRA fine-tuning via HuggingFace Trainer.

        Args:
            model: PEFT-wrapped model from `prepare()`.
            train_dataset: HuggingFace Dataset (or compatible) with
                tokenized inputs and labels matching the task type.
            eval_dataset: Optional evaluation dataset, same format.
            **kwargs: TrainingArguments overrides — learning_rate,
                num_epochs, per_device_batch_size, fp16, output_dir,
                data_collator, tokenizer.

        Returns:
            The trained model (same object as input, weights updated).

        Raises:
            ProcessingError: If transformers is missing or training fails.
        """
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError as e:
            raise ProcessingError(
                "transformers is required for training. "
                "Install with: pip install opentbtk[clinical_text]",
            ) from e

        training_args = TrainingArguments(
            output_dir=kwargs.get("output_dir", "./opentbtk_finetune_output"),
            learning_rate=kwargs.get("learning_rate", 2e-4),
            num_train_epochs=kwargs.get("num_epochs", 3),
            per_device_train_batch_size=kwargs.get("per_device_batch_size", 8),
            fp16=kwargs.get("fp16", False),
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            logging_steps=10,
            report_to=[],  # disable wandb/etc by default
        )

        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=kwargs.get("data_collator"),
                tokenizer=kwargs.get("tokenizer"),
            )
            trainer.train()
        except Exception as e:
            raise ProcessingError("Fine-tuning training run failed") from e

        log.info("finetuner.train.complete", output_dir=training_args.output_dir)
        return model

    def save(self, model: Any, output_path: str) -> None:
        """Save the LoRA adapter weights (not the full base model).

        Args:
            model: A trained PEFT model.
            output_path: Directory to save adapter weights and config.

        Raises:
            ProcessingError: If saving fails.
        """
        try:
            model.save_pretrained(output_path)
        except Exception as e:
            raise ProcessingError(
                f"Failed to save fine-tuned adapter to {output_path}",
                context={"output_path": output_path},
            ) from e
        log.info("finetuner.saved", output_path=output_path)
