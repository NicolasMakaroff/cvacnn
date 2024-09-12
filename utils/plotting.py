import wandb
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def image_prediction_logger(images, 
                            mask_pred, 
                            mask_th,
                            edt,
                            chan_vese,
                            attention
                            ):
    
    class_labels = {0: "healthy tissue", 1: "tumour"}

    #coord_barycenter = ((x1.squeeze().detach().numpy() + 1 )*64).astype(np.uint8)
    #coord_x0 = ((x0.squeeze() + 1 ) * 64).numpy().astype(np.uint8)
    #img = np.array(transforms.ToPILImage()(UnNormalize()(images.squeeze())))

    #img1 = cv2.circle(img=img, center=(coord_barycenter[1], coord_barycenter[0]), radius=3, color=(0, 255, 0), thickness=-1)
    #img2 = cv2.circle(img=img1, center=(coord_x0[1], coord_x0[0]), radius=3, color=(0, 0, 255), thickness=-1)

    wandb_image = wandb.Image(np.array(images), 
        masks={
            'predictions': {'mask_data': np.array(jnp.squeeze(mask_pred>0.5)), 'class_labels':class_labels},
            'ground_truth': {'mask_data': np.array(jnp.squeeze(mask_th >= 1)), 'class_labels':class_labels}
        },
        )

    wandb_prediction = wandb.Image(np.array(mask_pred))
    wandb_chanvese = wandb.Image(np.array(jnp.squeeze(chan_vese)))
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(np.array(jnp.squeeze(edt)))
    rgb_img = np.delete(rgba_img, 3, 2)
    wandb_edt = wandb.Image(rgb_img)

    wandb_att = wandb.Image(np.array(jnp.squeeze(attention)))

    return wandb_image, wandb_prediction, wandb_chanvese, wandb_edt, wandb_att
