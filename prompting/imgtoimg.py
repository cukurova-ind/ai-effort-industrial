import os
import time
import datetime
import tensorflow as tf

class Img2Img:
    def __init__(self, image_shape):

        self.output_channels = 3
        self.in_shape = image_shape

        self.generator_lr = 0.00004
        self.discriminator_lr = 0.00004

        self.generator = self.generator()
        self.discriminator = self.discriminator()

        self.checkpoint_dir = "img2img_checkpoints"
        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              discriminator=self.discriminator)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint,
                                            self.checkpoint_dir,
                                            max_to_keep=3)
                                            
        self.prev_epochs = 0

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_lr)
    
    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2,
                                padding="same",
                                kernel_initializer=initializer,
                                use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result
    
    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding="same",
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def generator(self):
        input_image = tf.keras.layers.Input(shape=self.in_shape)

        down_stack = [self.downsample(64, 4, apply_batchnorm=False),
                      self.downsample(128, 4),
                      self.downsample(256, 4),
                      self.downsample(512, 4),
                      self.downsample(512, 4),
                      self.downsample(512, 4),
                      self.downsample(512, 4),]

        up_stack = [self.upsample(512, 4, apply_dropout=True),
                    self.upsample(512, 4, apply_dropout=True),
                    self.upsample(512, 4, apply_dropout=True),
                    self.upsample(512, 4),
                    self.upsample(256, 4), 
                    self.upsample(128, 4),  
                    self.upsample(64, 4)]

        initializer = tf.random_normal_initializer(0., 0.02)

        x1 = input_image
        skips = []
        for down in down_stack:
            x1 = down(x1)
            skips.append(x1)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x1 = up(x1)
            x1 = tf.keras.layers.Concatenate()([x1, skip])

        y = tf.keras.layers.Conv2DTranspose(self.output_channels, 4,
                                            strides=2,
                                            padding="same",
                                            kernel_initializer=initializer,
                                            activation="tanh")(x1)

        model = tf.keras.Model(inputs=[input_image], outputs=y)

        return model
    
    @staticmethod
    def generator_loss(disc_generated_output, gen_output, target):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                         reduction=tf.keras.losses.Reduction.NONE)
        gan_loss = loss_object(tf.ones_like(disc_generated_output),
                            disc_generated_output)

        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (100 * l1_loss)

        return total_gen_loss
    
    def discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        input_image = tf.keras.layers.Input(shape=self.in_shape, name="input_image")
        target_image = tf.keras.layers.Input(shape=self.in_shape, name="target_image")

        x = tf.keras.layers.concatenate([input_image, target_image])

        down1 = self.downsample(64, 4, False)(x) 
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1) 
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)
        model = tf.keras.Model(inputs=[input_image, target_image], outputs=last)

        return model
    
    @staticmethod
    def discriminator_loss(disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                         reduction=tf.keras.losses.Reduction.NONE)
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = loss_object(tf.zeros_like(disc_generated_output),
                                    disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    def restore_model(self):
        if self.ckpt_manager.latest_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print ("Latest checkpoint restored!!")
            self.prev_epochs = (int(self.ckpt_manager.latest_checkpoint.split("-")[-1])*5)
    
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', tf.reduce_mean(gen_total_loss), step=step//1000)
            tf.summary.scalar('disc_loss', tf.reduce_mean(disc_loss), step=step//1000)

    def fit(self, train_ds, test_ds, steps):

        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                pass

                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()

                prediction = self.generator(example_input, training=True)
                for i, images in enumerate(zip(example_input, example_target, prediction)):
                    tf.keras.utils.save_img("input"+str(i)+".png", images[0] * 127.5 + 127.5)
                    tf.keras.utils.save_img("truth"+str(i)+".png", images[1] * 127.5 + 127.5)
                    tf.keras.utils.save_img("pred"+str(i)+".png", images[2] * 0.5 + 0.5)
                print(f"Step: {step//1000}k")

            self.train_step(input_image, target, step)
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)
            if (step+1) % 5000 == 0:
                self.ckpt_manager.save()
    
    def train(self, train_ds, val_ds, num_epochs=100, retrain=False):

        if retrain:
            self.restore_model()
        else:
            folder = self.checkpoint_dir
            for fn in os.listdir(folder):
                fp = os.path.join(folder, fn)
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_lr)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_lr)

        generator_loss_log = []
        discriminator_loss_log = []
        steps = 20
        batch_iter = iter(train_ds)
        for epoch in range(self.prev_epochs, num_epochs + self.prev_epochs):
            print("Epoch %d/%d:\n ["%(epoch + 1, num_epochs + self.prev_epochs), end = "")
            start_time = time.time()

            for step in range(steps):
                if step % 5 == 0:
                    print("=", end="", flush=True)
                image_batch, target_batch = next(batch_iter)
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    gen_output = self.generator(image_batch, training=True)

                    disc_real_output = self.discriminator([image_batch, target_batch],
                                                            training=True)
                    disc_generated_output = self.discriminator([image_batch, gen_output],
                                                                training=True)
                    gen_total_loss = self.generator_loss(
                        disc_generated_output, gen_output, target_batch)
                    disc_loss = self.discriminator_loss(
                        disc_real_output, disc_generated_output)
                    
                generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
                discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

                generator_loss_log.append(gen_total_loss)
                discriminator_loss_log.append(disc_loss)

            end_time = time.time()
            if epoch % 1 == 0:
                epoch_time = end_time - start_time
                template = "] - generator_loss: {:.4f} - discriminator_loss: {:.4f} - epoch_time: {:.2f} s"
                print(template.format(tf.reduce_mean(generator_loss_log), tf.reduce_mean(discriminator_loss_log), epoch_time))

            if (epoch + 1) % 5 == 0 or epoch==num_epochs-1:

                example_input, example_target = next(iter(val_ds.take(1)))
                prediction = self.generator(example_input, training=True)
                for i, images in enumerate(zip(example_input, example_target, prediction)):
                    tf.keras.utils.save_img("input"+str(i)+".png", images[0] * 127.5 + 127.5)
                    tf.keras.utils.save_img("truth"+str(i)+".png", images[1] * 127.5 + 127.5)
                    tf.keras.utils.save_img("pred"+str(i)+".png", images[2] * 0.5 + 0.5)

                self.ckpt_manager.save()