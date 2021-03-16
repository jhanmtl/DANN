import tensorflow as tf
import numpy as np


@tf.custom_gradient
def reverse_gradient(x):
    def grad(dy):
        return -1 * dy

    return x, grad


class GRL(tf.keras.layers.Layer):
    def __init__(self):
        super(GRL, self).__init__()

    def call(self, x):
        return reverse_gradient(x)


class DANN():
    def __init__(self, batchsize=1, epochs=15, alpha=5, classifier_lr=1e-3, extractor_lr=1e-3, discriminator_lr=1e-3):
        super(DANN, self).__init__()

        self.alpha = alpha
        self.batchsize = batchsize
        self.epochs = epochs

        self.classifier_lr = classifier_lr
        self.extractor_lr = extractor_lr
        self.discriminator_lr = discriminator_lr

        self.classifier_opt = tf.keras.optimizers.Adam(self.classifier_lr)
        self.extractor_opt = tf.keras.optimizers.Adam(self.extractor_lr)
        self.discriminator_opt = tf.keras.optimizers.Adam(self.discriminator_lr)

        self.cce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()

        self.class_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.class_loss = tf.keras.metrics.Mean()

        self.domain_acc = tf.keras.metrics.BinaryAccuracy()
        self.domain_loss = tf.keras.metrics.Mean()

        self.extractor = None
        self.classifier = None
        self.discriminator = None

        self.domain_predictor = None
        self.class_predictor = None

    def set_submodels(self, submodel_dict):
        self.extractor = submodel_dict['extractor']
        self.classifier = submodel_dict['classifier']
        self.discriminator = submodel_dict['discriminator']

        self.class_predictor = tf.keras.models.Sequential([self.extractor, self.classifier])
        self.domain_predictor = tf.keras.models.Sequential([self.extractor, self.discriminator])

    @tf.function
    def _cross_domain_train_step(self, source_x, source_y, target_x):

        self._reset_metrics()
        domain_labels, domain_x = self._generate_domain_training_data(source_x, target_x)

        with tf.GradientTape(persistent=True) as tape:
            class_pred = self.class_predictor(source_x, training=True)
            class_loss = self.cce_loss(source_y, class_pred)

            domain_pred = self.domain_predictor(domain_x, training=True)
            domain_loss = self.bce_loss(domain_labels, domain_pred)
            domain_loss *= self.alpha

        classifier_gradient = tape.gradient(class_loss, self.class_predictor.trainable_variables)
        extractor_gradient_by_class_loss = tape.gradient(class_loss, self.extractor.trainable_variables)
        extractor_gradient_by_domain_loss = tape.gradient(domain_loss, self.extractor.trainable_variables)
        discriminator_gradient = tape.gradient(domain_loss, self.discriminator.trainable_variables)

        self.classifier_opt.apply_gradients(zip(classifier_gradient, self.class_predictor.trainable_variables))
        self.extractor_opt.apply_gradients(zip(extractor_gradient_by_domain_loss, self.extractor.trainable_variables))
        self.extractor_opt.apply_gradients(zip(extractor_gradient_by_class_loss, self.extractor.trainable_variables))
        self.discriminator_opt.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        self.class_loss(class_loss)
        self.class_acc(source_y, class_pred)

        self.domain_loss(domain_loss)
        self.domain_acc(domain_labels, domain_pred)

    def cross_domain_train(self, seen_x, seen_y, unseen_x):

        TRAIN_STEPS_PER_EPOCH = min(int(len(seen_x) / self.batchsize),int(len(unseen_x) / self.batchsize))+1

        seen_dataset = tf.data.Dataset.from_tensor_slices((seen_x, seen_y))
        seen_dataset = seen_dataset.shuffle(buffer_size=len(seen_x) + 1).batch(self.batchsize)

        unseen_dataset = tf.data.Dataset.from_tensor_slices(unseen_x)
        unseen_dataset = unseen_dataset.shuffle(buffer_size=len(unseen_x) + 1).batch(self.batchsize)

        for epoch in range(self.epochs):
            print("epoch {}/{}".format(epoch + 1, self.epochs))
            tbar = tf.keras.utils.Progbar(target=TRAIN_STEPS_PER_EPOCH, unit_name="batch", width=30)

            for count, ((source_x, source_y), target_x) in enumerate(zip(seen_dataset, unseen_dataset)):
                self._cross_domain_train_step(source_x, source_y, target_x)

                progbar_classifier_acc = self.class_acc.result().numpy()
                progbar_classifier_loss = self.class_loss.result().numpy()

                progbar_domain_acc = self.domain_acc.result().numpy()
                progbar_domain_loss = self.domain_loss.result().numpy()

                progbar_values = [('classifier acc', progbar_classifier_acc),
                                  ('classifier loss', progbar_classifier_loss),
                                  ('domain acc', progbar_domain_acc),
                                  ('domain_loss', progbar_domain_loss)]

                tbar.update(count, values=progbar_values)
            tbar.update(count + 1, values=progbar_values, finalize=True)

    def source_domain_train(self, source_x, source_y):

        self.class_predictor.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                     optimizer=tf.keras.optimizers.Adam(self.classifier_lr),
                                     metrics=['acc'])

        self.class_predictor.fit(source_x,
                                 source_y,
                                 batch_size=self.batchsize,
                                 epochs=self.epochs)

    def classify(self, x):
        pred_scores = self.class_predictor.predict(x)
        pred_classes = np.argmax(pred_scores, axis=-1)
        return pred_classes

    def evaluate_classifier_acc(self, x, ytrue):
        ypred = self.classify(x)
        acc = np.mean(ypred == ytrue)
        return acc

    def _reset_metrics(self):
        self.class_acc.reset_states()
        self.class_loss.reset_states()
        self.domain_acc.reset_states()
        self.domain_loss.reset_states()

    def _generate_domain_training_data(self, source_x, target_x):
        source_domain_label = np.zeros(len(source_x))
        target_domain_label = np.ones(len(target_x))
        domain_labels = np.concatenate((source_domain_label, target_domain_label))
        domain_labels = tf.convert_to_tensor(domain_labels, dtype=tf.int64)

        combined_x = tf.concat((source_x, target_x), axis=0)

        shuffle_idx = tf.range(len(domain_labels))
        shuffle_idx = tf.random.shuffle(shuffle_idx)

        domain_labels = tf.gather(domain_labels, shuffle_idx)
        combined_x = tf.gather(combined_x, shuffle_idx, axis=0)

        return domain_labels, combined_x


def submodel_config_march_04(num_class):
    extractor = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(100, 3)),
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu')
    ])

    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_class, activation='softmax')
    ])

    discriminator = tf.keras.models.Sequential([
        GRL(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    m_dict = {"extractor": extractor,
              "classifier": classifier,
              "discriminator": discriminator}

    return m_dict
