config = {
    "crop_size": 128,
    "sr_factor": 2,
    "upsample": "cubic", # available values pixelshuffle, cubic
    "number_of_iterations": 15000,
    "loss_type": "l1",
    "l1_loss_coff": 0.1,
    "content_loss_coff": 0.9,
    "device": "cuda",
    "learning_rate": 0.00001,
    "model": "zssr",
    "metrics": ["psnr"]
}