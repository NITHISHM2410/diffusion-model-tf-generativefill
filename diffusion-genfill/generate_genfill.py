import tensorflow as tf
from tqdm import tqdm


class GenerateImagesGenFill:
    def __init__(self, model, device):
        """
        :param model: A trained GenFill diffusion model instance
        :param device: A tf.distribute.Strategy instance
        """
        self.model = model
        self.device = device

    @tf.function
    def distribute_fn(self, fn, args):
        output = self.device.run(fn, args=args)
        return output

    def sample(self, no_of, image_references):
        """
        Generates images from masked image references.
        :param no_of:  Number of images to generate. Usually 64-TPU & 16-GPU.
        :param image_references: Sample images that undergoes masking which the generative model unmasks using
        generative fill. To change the mask percent, update the mask percent in 'UNetGenFill' model and initialize
        a new instance of this class.

        :return: Masked images, Unmasked images
        """
        # Sample Gaussian noise
        images = tf.random.normal((no_of, self.model.img_res, self.model.img_res, 3))
        masked_images = self.model.mask_out(image_references)

        images = self.device.experimental_distribute_dataset(
            tf.data.Dataset.from_tensor_slices(images).batch(no_of, drop_remainder=True)
        )
        images = next(iter(images))

        masked_images = self.device.experimental_distribute_dataset(
            tf.data.Dataset.from_tensor_slices(masked_images).batch(no_of, drop_remainder=True)
        )
        masked_images = next(iter(masked_images))

        # Reverse diffusion for t time
        for t in tqdm(reversed(tf.range(0, self.model.time_steps)), "Sampling Images...", self.model.time_steps):
            images = self.distribute_fn(self.model.diffuse_step, (images, t, masked_images))

        images = self.device.experimental_local_results(images)
        images = tf.concat(images, axis=0)

        masked_images = self.device.experimental_local_results(masked_images)
        masked_images = tf.concat(masked_images, axis=0)

        # Set pixel values in display range
        images = tf.clip_by_value(images, 0, 1)
        return masked_images, images


def read_images(image_locations, size, return_ds=False):
    """
    Read images from locations and applies basic transformations

    :param image_locations: list of image locations
    :param size: required image size
    :param return_ds: boolean value, whether to return images as tf.data.Dataset
    """
    image_locations = tf.data.Dataset.from_tensor_slices(image_locations)

    def transform(img):
        img = tf.io.decode_jpeg(tf.io.read_file(img), channels=3)
        img = tf.image.resize(img, (size, size))
        img = img / 255
        return img

    images = image_locations.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    if return_ds:
        return images

    images = images.batch(len(image_locations), drop_remainder=True)
    return next(iter(images))



