import numpy as np
from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Dense,LeakyReLU,Activation,\
            Conv2D,Conv2DTranspose,Flatten,Reshape,BatchNormalization
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import Adam

try:
    from tensorflow.keras.utils import plot_model
except Exception as e:
    print(e)

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255


class MnistGAN:
    def __init__(self,gen_lr=0.0002,dis_lr=0.0002):
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.genModel = self.build_generator()
        self.disModel = self.build_discriminator()
        self.advModel = self.build_adversialmodel()
    def generator_block(self,input,u,k,s=2,p='same',a='leakyrelu',bn=True):
        x = Conv2DTranspose(u,kernel_size=k,strides=s,padding=p)(input)
        if a=='leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif a=='sigmoid':
            x = Activation('sigmoid')(x)
        elif a=='tanh':
            x = Activation('tanh')(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    def discriminator_block(self,input,u,k,s=2,p='padding',a='leakyrelu'):
        x = Conv2D(u,kernel_size=k,strides=s,padding=p)(input)
        if a=='leakyrelu':
            x = LeakyReLU(0.2)(x)
        elif a=='sigmoid':
            x = Activation('sigmoid')(x)
        return x
    def build_generator(self):
        input_layer = Input(shape=(100, ))
        x = Dense(7*7*32)(input_layer)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Reshape((7, 7, 32))(x)
        x = self.generator_block(x, u=128, k=2, s=2, p='same')
        x = self.generator_block(x, u=256, k=2, s=2, p='same')
        out = self.generator_block(x,u=1,k=1,s=1,p='same',a='sigmoid',bn=False)

        opt=Adam(learning_rate=self.gen_lr,beta_1=0.5)
        model = Model(inputs=input_layer,outputs=out)
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer=opt)
        plot_model(model,to_file='gen.png',show_shapes=True)
        return model

    def build_discriminator(self):
        input_layer = Input(shape=(28, 28, 1))
        x = self.discriminator_block(input=input_layer,u=256,k=2,s=2,p='same',a='leakyrelu')
        x = self.discriminator_block(input=x, u=128, k=2, s=2, p='same', a='leakyrelu')
        x = Flatten()(x)
        x = Dense(64)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1)(x)
        out = Activation('sigmoid')(x)

        opt = Adam(learning_rate=self.dis_lr,beta_1=0.5)
        model = Model(inputs=input_layer,outputs=out)
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        plot_model(model, to_file='dis.png', show_shapes=True)
        return model

    def build_adversialmodel(self):
        z_noise_input = Input(shape=(100, ))
        gen_sample = self.genModel(z_noise_input)
        self.disModel.trainable = False
        out = self.disModel(gen_sample)
        opt = Adam(learning_rate=self.gen_lr,beta_1=0.5)
        model = Model(inputs=z_noise_input, outputs=out)
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer=opt)
        return model
    def train(self,epochs=30000):
        batch_size = 32
        dloss = []
        gloss = []
        for i in range(epochs):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            noise = np.random.normal(0, 1, [batch_size, 100])
            fake_images = self.genModel(noise)
            real_images = x_train[idx]

            y_real = np.ones((batch_size, 1))
            y_fake = np.zeros((batch_size, 1))

            d_loss_fake = self.disModel.train_on_batch(x=fake_images, y=y_fake)
            d_loss_real = self.disModel.train_on_batch(x=real_images, y=y_real)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            dloss.append(d_loss[0])
            noise = np.random.normal(0, 1, [batch_size, 100])
            g_loss = self.advModel.train_on_batch(noise, y_real)
            g_loss2 = self.advModel.train_on_batch(noise, y_real)
            g_loss3 = self.advModel.train_on_batch(noise, y_real)
            gloss.append((g_loss+g_loss2+g_loss3)/3)
            print(f"epochs:{i+1} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.3f}] [G loss: {g_loss}]")
            if (i+1)%500==0 or i==0:
                num_images=16
                noise = np.random.normal(0,1,[num_images, 100])
                generated_images = self.genModel(noise)
                # generated_images = 0.5 * generated_images + 0.5
                plt.figure(figsize=(8, 8))
                plt.title(f'{i+1} epochs train')
                for j in range(num_images):
                    plt.subplot(4, 4, j + 1)
                    plt.imshow(generated_images[j, :, :, 0], cmap="gray")
                    plt.axis("off")
                plt.savefig(f"images/epoch{i+1}.png")

        self.genModel.save('generator2.h5')
        self.disModel.save('discriminator2.h5')
        self.advModel.save('adversarial2.h5')
        plt.clf()
        plt.ylim(-0.1,1)
        plt.title('loss')
        plt.plot(gloss,label='g')
        plt.plot(dloss,label='d')
        plt.legend(loc='best')
        plt.savefig(f"loss.png")

def predict():
    num_images = 25
    noise = np.random.normal(0, 1, [num_images, 100])
    model = load_model('generator2.h5')
    generated_images = model(noise)
    plt.figure(figsize=(8, 8))
    plt.title(f'predict')
    for j in range(num_images):
        plt.subplot(5, 5, j + 1)
        plt.imshow(generated_images[j, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.show()

def showinput():
    num_images = 25
    noise = np.random.normal(0, 1, [num_images, 100])
    noise = noise.reshape((-1,10,10,1))
    plt.figure(figsize=(8, 8))
    plt.title(f'predict')
    for j in range(num_images):
        plt.subplot(5, 5, j + 1)
        plt.imshow(noise[j, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.show()
if __name__=='__main__':
    g=MnistGAN()
    g.train(epochs=50000)
    # for _ in range(2):
    #     showinput()
    #     predict()

