from typing import Dict, Union, Any

import torch
from torch import nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled, is_torch_xpu_available, is_torch_mlu_available, \
    is_torch_musa_available, is_torch_npu_available, is_torch_mps_available, is_apex_available
from transformers.trainer_pt_utils import smp_forward_backward
from transformers.training_args import OptimizerNames

if is_apex_available():
    from apex import amp


class myTrainer(Trainer):
    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
                Perform a training step on a batch of inputs.

                Subclass and override to inject custom behavior.

                Args:
                    model (`nn.Module`):
                        The model to train.
                    inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                        The inputs and targets of the model.

                        The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                        argument `labels`. Check your model's documentation for all accepted arguments.

                Return:
                    `torch.Tensor`: The tensor with training loss on this batch.
                """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()
