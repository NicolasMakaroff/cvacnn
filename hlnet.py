import jax
import jax.numpy as jnp
import haiku as hk
from utils.nets_att import UNet
from utils.losses_metrics import bce_dice_loss, compute_iou, hausdorff, F1, precision_recall_FP_FN
from utils.plotting import image_prediction_logger
import optax
import wandb
from utils.trainer import TrainerModule, TrainState
import matplotlib.pyplot as plt
import utils.transforms as transforms
import numpy as np


class UNetTrainer(TrainerModule):
    def __init__(self, 
                num_classes: int,
                **kwargs):
        super().__init__(model_name='CVACNN',
                        model_class=UNet,
                        model_hparams={
                            'num_classes': num_classes,
                            'use_batch_norm': True,
                            'bn_config' : {'decay_rate': 0.9},
                        },
                         **kwargs)

    def create_functions(self):
        def loss_fn(params, state, loss_scale, rng, batch, is_training):
            output = self.forward.apply(params, state, None, batch, is_training=is_training)
            logits, state = output if is_training else output
            images, mask = batch
            loss = bce_dice_loss(logits[0], mask).mean()
            metrics = {'loss': loss}
            metrics['iou'] = compute_iou(logits[0], mask).mean()
            #if not is_training:
                #self.logging(images, mask, logits)
            return loss_scale.scale(loss), (metrics, state)

        #@functools.partial(jax.pmap, axis_name='i')
        def train_step(train_state, batch):
            params, state, opt_state, loss_scale, rng = train_state
            grads, (metrics, new_state) = (
                jax.grad(loss_fn, has_aux=True)(params, state, loss_scale, rng, batch, is_training=True))
            #jax.debug.print('grads min: {x}', x=min(grads))
            #jax.debug.print('grads max: {x}', x=max(grads))
            #jax.debug.print('grads shape: {x}', x=grads.shape)
            # Grads are in "param_dtype" (likely F32) here. We cast them back to the
            # compute dtype such that we do the all-reduce below in the compute precision
            # (which is typically lower than the param precision).
            policy = self.precision_policy()
            grads = policy.cast_to_compute(grads)
            grads = loss_scale.unscale(grads)


            # We compute our optimizer update in the same precision as params, even when
            # doing mixed precision training.
            grads = policy.cast_to_param(grads)

            # Compute and apply updates via our optimizer.
            updates, new_opt_state = self.optimizer().update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            self.state = TrainState(new_params, new_state, new_opt_state, loss_scale, rng=rng)
            return self.state, metrics, grads

        def eval_step(train_state, batch):
            _, (metrics, _) = loss_fn(train_state.params, train_state.state, train_state.loss_scale, train_state.rng, batch, is_training=False)
            return metrics

        return train_step, eval_step

    def on_validation_epoch_end(self, epoch_idx, eval_metrics, val_loader):
        batch = next(iter(val_loader))
        output = self.forward.apply(self.state.params, self.state.state, None, batch, is_training=False)
        logits, _ = output
        images, mask = batch
        hausdorff_distance, f1_score = 0, 0, 
        precision = [0, 0, 0, 0]
        for mask_pred, mask_init in zip(logits[0], mask):
            hausdorff_distance += hausdorff(mask_pred, mask_init >= 1. * 1.)
            f1_score += F1(mask_pred, mask_init >= 1. * 1.)
            precision = [np.mean(i) for i in zip(precision,  precision_recall_FP_FN(mask_pred, mask_init >= 1. * 1.))] 
        wandb.log({'val/hausdorff': hausdorff_distance/logits[0].shape[0], 'val/F1': f1_score/logits[0].shape[0], 'val/precision': precision[0], 'val/recall': precision[1], 'val/FNR': precision[2], 'val/FPR': precision[3]})
        self.logging(images, mask, logits, epoch_idx)
        self.save_model()


    def run_model_init(self, exmp_input, init_rng):
        rng = jax.random.PRNGKey(self.seed)
        return self.forward.init(rng, exmp_input, is_training=True)

    def logging(self, images, mask, logits, epoch_idx):
        wandb_seg = []
        wandb_pred = []
        wandb_chan_vese = []
        wandb_euclidian_distance = []
        wandb_attention = []
        for idx, (img, mask, prediction, edt, chan_vese, attention) in enumerate(zip(images, mask, logits[0], logits[1], logits[2], logits[3])):
            segmentation = prediction
            wandb_segmentation, wandb_prediction, wandb_chan, wandb_edt, wandb_att = image_prediction_logger(img, segmentation, mask, edt, chan_vese, attention)
            wandb_seg.append(wandb_segmentation)
            wandb_pred.append(wandb_prediction)
            wandb_chan_vese.append(wandb_chan)
            wandb_euclidian_distance.append(wandb_edt)
            wandb_attention.append(wandb_att)

            #np.save('./array_rectangle/prediction.npy', img)
            #np.save('./array/edt.npy', edt)
            #if idx in [1, 9, 14, 22, 31]:
                #np.save('./array_final/chan_vese_epoch-'+ str(epoch_idx)+ '_idx-' + str(idx) + '.npy', chan_vese)
        wandb.log({'Tumour Segmentation':wandb_seg[0:10], 
                    'Prediction':wandb_pred[0:10], 
                    'Chan Vese':wandb_chan_vese[0:10], 
                    'Euclidian Distance Transform':wandb_euclidian_distance[0:10], 
                    'Attention':wandb_attention[0:10]})



        
