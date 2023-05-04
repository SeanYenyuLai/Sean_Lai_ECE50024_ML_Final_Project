from keras.layers import Dense, Dropout, Input, Conv2D, Conv2DTranspose,\
    Flatten, Embedding, Reshape, Concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
"""Data"""

(x_t, y_t), (X_TE, Y_TE) = mnist.load_data()
x_t = x_t.astype('float32')
x_t = np.expand_dims(x_t, -1)
x_t = x_t / 255

"""Discriminator"""
def discriminator(layers, opt, in_shape=(28, 28, 1), n_classes=10):

    labels_in = Input(shape=(1,))
    di_embd = Embedding(n_classes, 50)(labels_in)
    n = in_shape[0] * in_shape[1]
    dense_li = Dense(n)(di_embd)
    resh = Reshape((in_shape[0], in_shape[1], 1))(dense_li)
    im_in = Input(shape=in_shape)
    Y = Concatenate()([im_in, resh])
    for i in range(layers):
        Y = Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape)(Y)
        Y = LeakyReLU()(Y)
        Y = Dropout(0.4)(Y)
    out_f = Flatten()(Y)
    out_f = Dense(1, activation='sigmoid')(out_f)
    model = Model([im_in, labels_in], out_f)
    #Optimizer
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

"""Generator"""
def generator(layers, latent_dim, n_classes=10):
    in_label = Input((1,))
    ge_embd = Embedding(n_classes, 50)(in_label)
    nodes = 7 * 7
    den_label = Dense(nodes)(
        ge_embd)
    resh = Reshape((7, 7, 1))(den_label)
    la_v = Input(shape=(latent_dim,))
    nodes = 128 * 7 * 7

    out = Dense(nodes)(la_v)
    out = LeakyReLU()(out)
    out = Reshape((7, 7, 128))(out)

    Y = Concatenate()([out, resh])
    for i in range(layers):
        Y = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(Y)
        Y = LeakyReLU()(Y)
        Y = Dropout(0.4)(Y)


    out_f = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(Y)
    model = Model([la_v, in_label], out_f)
    return model


"""GAN"""
def Gan(gen, dis, opt):
    dis.trainable = False
    gen_noise, gen_label = gen.input
    gen_output = gen.output
    gan_output = dis([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gan_output)
    #model
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def Model_Training(epochs, batch_per_epoch, bat_g_d, d, loss_dis_real, latent_dim, loss_dis_fake, g, bat_size, gan, WL, loss_g):
    for epoch in range(epochs):
        x = 0
        for _ in tqdm(range(batch_per_epoch)):

            #Real Image
            indices = np.random.randint(0, x_t.shape[0], bat_g_d)
            x_real = x_t[indices]
            label_real = y_t[indices]
            y_real = np.ones((bat_g_d, 1))
            dLReal, _ = d.train_on_batch([x_real, label_real], y_real)
            loss_dis_real.append(dLReal)

            #Fake Image
            x_fake = np.random.randn(latent_dim * bat_g_d)
            x_fake = x_fake.reshape(bat_g_d, latent_dim)
            label_fake = np.random.randint(0, 10, bat_g_d)
            x_fake_img = g.predict([x_fake, label_fake])  # Generator => Fake Image
            y_fake = np.zeros((bat_g_d, 1))
            dLFake, _ = d.train_on_batch([x_fake_img, label_fake], y_fake)
            loss_dis_fake.append(dLFake)

            # dis_performance = d.predict([x_fake_img, y_real])
            # print(float(sum(dis_performance)/len(dis_performance)))

            #GAN
            x_fake_g = np.random.randn(latent_dim * bat_size)
            label_gan = np.random.randint(0, 10, bat_size)
            x_fake_g = x_fake_g.reshape(bat_size, latent_dim)
            y_gan = np.ones((bat_size, 1))
            g_l = gan.train_on_batch([x_fake_g, label_gan], y_gan)

            ##########################
        l_V = np.random.randn(latent_dim * 100)
        l_V = l_V.reshape(100, latent_dim)
        label_t = np.array([x for _ in range(10) for x in range(10)])
        gen_im = g.predict([l_V, label_t])
        dis_per = d.predict([gen_im, label_t])
        print(dis_per)
        for k in dis_per:
            if float(k) < 0.5:
                x += 1
        win = float(x / 100)
        print(sum(dis_per)/100)
        print(float(x / 100))
        WL.append(win)



        if epoch % 5 == 0:
            l_V = np.random.randn(latent_dim * 100)
            l_V = l_V.reshape(100, latent_dim)
            label_t = np.array([x for _ in range(10) for x in range(10)])

            # gen_im = g.predict([l_V, label_t])
            # dis_per = d.predict([gen_im, label_t])
            # x =0
            # for k in dis_per:
            #     if k > 0.4:
            #         x += 1
            #         #############################
            # xx.append(x/len(dis_per))
                ##############################
            count = 0
            fig, ax = plt.subplots(10, 10)
            for i in range(10):
                for j in range(10):
                    ax[i][j].axis('off')
                    ax[i][j].imshow(np.squeeze(gen_im[count], axis=-1), cmap='gray')
                    count += 1
            fig.savefig(f'gen_images{epoch}.jpg')
            loss_g.append(g_l)
    return WL

def Visualize(WL, WL_moving_ave, loss_dis_real, loss_dis_fake, loss_g):
    fig, axs = plt.subplots()
    plt.plot(WL)
    fig.savefig(f'win_loss.jpg')
    with open('win_loss.txt', 'w') as f:
        for i in WL:
            f.write(str(i))
            f.write('\n')
    plt.close(fig)

    fig, axs = plt.subplots()
    plt.plot(WL_moving_ave)
    fig.savefig(f'win_loss_moving_ave.jpg')
    plt.close(fig)

    fig, axs = plt.subplots(3)
    fig.suptitle('Loss Function')
    axs[0].plot(loss_dis_real)
    plt.plot(loss_dis_real)
    axs[1].plot(loss_dis_fake)
    plt.plot(loss_dis_fake)
    axs[2].plot(loss_g)
    plt.plot(loss_g)
    plt.ylabel('Loss GAN')
    fig.savefig(f'loss_images.jpg')

    # def generate_and_visualize():
    #     l_V = np.random.randn(latent_dim * 100)
    #     l_V = l_V.reshape(100, latent_dim)
    #     t_label = np.array([x for _ in range(10) for x in range(10)])
    #     gen_im = g.predict([l_V, t_label])
    #
    #     count = 0
    #     fig, ax = plt.subplots(10, 10)
    #     for i in range(10):
    #         for j in range(10):
    #             ax[i][j].axis('off')
    #             ax[i][j].imshow(np.squeeze(gen_im[count], axis=-1), cmap='gray')
    #             count += 1
    #     plt.show()
    #
    #
    # generate_and_visualize()


def main():

    """Basic parameters"""
    bat_size = 256
    bat_g_d = 128
    epochs = 300
    latent_dim = 100
    batch_per_epoch = x_t.shape[0] // bat_size

    """Models"""
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    d = discriminator(layers=2, opt=optimizer)
    g = generator(layers=2, latent_dim=latent_dim)
    gan = Gan(g, d, optimizer)

    """Train"""
    loss_dis_real = []
    loss_dis_fake = []
    loss_g = []
    WL = []

    """Training"""
    WL = Model_Training(epochs, batch_per_epoch, bat_g_d, d, loss_dis_real, latent_dim, loss_dis_fake, g, bat_size, gan, WL, loss_g)

    ar = WL
    window_size = 10
    i = 0
    WL_moving_ave = []

    while i < len(ar) - window_size + 1:
        window_ave = round(np.sum(ar[i:i + window_size]) / window_size, 2)
        WL_moving_ave.append(window_ave)
        i += 1

    """Visualize"""
    Visualize(WL, WL_moving_ave, loss_dis_real, loss_dis_fake, loss_g)


if __name__ == '__main__':
    main()