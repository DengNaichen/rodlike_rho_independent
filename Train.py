import matplotlib.pyplot as plt
from MyLossFunction import cross_entropy_loss, free_energy
import torch


def fit(x, y, net, epochs, learning_rate, diagram=False):
    if diagram:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.subplots_adjust(left=0.15, bottom=0.1, top=0.9, right=0.95, hspace=0.4, wspace=0.25)

    loss_points = []
    for i in range(epochs):
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        yhat = net(x)
        loss = cross_entropy_loss(yhat, y)
        loss_points.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and diagram:
            ax1.cla()
            ax2.cla()
            ax1.scatter(x.data.numpy(), y.data.numpy(), marker=".")
            ax1.plot(x.data.numpy(), yhat.data.numpy(), 'r-', lw=1)
            ax1.set_xlabel(" \\theta")
            ax2.set_xlim([0, epochs])
            ax2.text(150, 0.5, 'Loss=%.5f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'black'})
            ax2.text(300, 0.5, 'epochs=%i' % i, fontdict={'size': 10, 'color': 'blue'})
            ax2.plot(range(epochs)[:i], loss_points[:i])
            ax2.set_xlabel("epochs")
            ax2.set_ylabel("Loss")
            plt.pause(0.1)
    if diagram:
        plt.ioff()
        plt.show()
    return net


def train_free_energy(net, x, rho, lambd, optimizer, matrix, epochs, diagram=True):
    if diagram:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.subplots_adjust(left=0.15, bottom=0.1, top=0.9, right=0.95, hspace=0.4, wspace=0.25)
    loss_points = []

    for i in range(epochs):
        yhat = net(x)
        loss = free_energy(yhat, x, matrix, rho, lambd)
        loss_points.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0 and diagram == True:
            ax1.cla()
            ax2.cla()
            ax1.plot(x.data.numpy(), yhat.data.numpy())
            ax1.set_xlabel("$\\theta$")
            ax2.set_ylabel("$f$")
            ax1.set_title("Distribution Function")
            ax2.set_xlim([0, epochs])
            ax2.text(epochs / 4, 1.1, 'Loss=%.7f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'black'})
            ax2.text(epochs / 1.5, 1.1, 'epochs=%i' % i, fontdict={'size': 10, 'color': 'blue'})
            ax2.set_xlabel("epochs")
            ax2.set_ylabel("Loss(Free Energy)")
            ax2.set_title("Free Energy with $\\rho =$ {} and $\\lambda = $ {}".format(rho, lambd))
            ax2.plot(range(epochs)[:i], loss_points[:i])
            plt.pause(0.1)
    if diagram:
        plt.ioff()
        plt.show()

    return net


# why we need tensor data: we only want to use the "tensor" data in tensor without gradient data,
# .data can do this work


def train_free_energy_two_inputs(net, x, inputs, rho, lambd, scheduler, simpson_matrix,
                                 optimizer, matrix, epochs, diagram=True):
    """
    :param net: NN with random parameters
    :param x: an array, from 0 to 2pi
    :param inputs: inputs is an matrix with dimension (len(x), 2), sin(x) and cos(x)
    :param rho: constant
    :param lambd: penalty norm
    :param optimizer: the optimizer
    :param scheduler: schedule learning rate
    :param matrix: |sin(theta - theta')|
    :param epochs: number of iterations
    :param diagram: boolean, weather to plot (and save) the diagram
    :return: NN with trained parameters
    """

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.subplots_adjust(left=0.15, bottom=0.1, top=0.9, right=0.95, hspace=0.4, wspace=0.25)
    loss_points = []

    for i in range(epochs):
        yhat = net(inputs) # the different from previous one
        # todo: try to control some data points
        # yhat_clone = yhat.clone()
        # yhat_clone[0] = yhat_clone[1]
        # yhat_clone[-1] = yhat_clone[-2]
        # todo: end
        # loss = free_energy(yhat_clone, x, matrix, rho, lambd)
        loss = free_energy(yhat, rho, x, simpson_matrix, matrix, lambd)
        loss_points.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 500 == 0:
            # plot the output
            ax1.cla()
            ax1.plot(x.data.numpy(), yhat.data.numpy())
            ax1.scatter(x.data.numpy()[0], yhat.data.numpy()[0])
            ax1.scatter(x.data.numpy()[len(x)-1], yhat.data.numpy()[len(x)-1])
            ax1.scatter(x.data.numpy()[1], yhat.data.numpy()[1])
            ax1.scatter(x.data.numpy()[len(x) - 2], yhat.data.numpy()[len(x) - 2])
            ax1.annotate('(%.2f, %.4f)' % (x.data.numpy()[0], yhat.data.numpy()[0]),
                         (x.data.numpy()[0], yhat.data.numpy()[0]))
            ax1.annotate('(%.2f, %.4f)' % (x.data.numpy()[len(x)-1], yhat.data.numpy()[len(x)-1]),
                         (x.data.numpy()[len(x)-1], yhat.data.numpy()[len(x)-1]))
            ax1.set_xlabel("$\\theta$")
            ax1.set_ylabel("$f$")
            ax1.set_title("Distribution Function")
            # plot the cost
            ax2.cla()
            ax2.set_xlim([0, epochs])
            ax2.text(epochs / 4, 1.1, 'Loss=%.7f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'black'})
            ax2.text(epochs / 1.5, 1.1, 'epochs=%i' % i, fontdict={'size': 10, 'color': 'blue'})
            ax2.set_xlabel("epochs")
            ax2.set_ylabel("Loss(Free Energy)")
            ax2.set_title("Free Energy with $\\rho =$ {} and $\\lambda = $ {}".format(rho, lambd))
            ax2.plot(range(epochs)[:i], loss_points[:i])
            # 0.1 second between frames
            plt.pause(0.1)

    if diagram:
        plt.ioff()
        plt.show()
        plt.savefig('figure/' + 'rho=' + str(rho) + "lambda=" + str(lambd) + '.pdf')

    return net

