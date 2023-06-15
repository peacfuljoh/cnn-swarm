"""Simple CNNs for grayscale and RGB images (CIFAR10, LeNet)"""

from nn_utils import eval_classifier_test_acc
from ml_ops import train, get_cifar10_data, init_net



def try_net_on_cifar10():
    # get CIFAR10 dataset
    batch_size = 64
    trainloader, testloader, _ = get_cifar10_data(batch_size=batch_size)

    # visualize batch
    if 0:
        vis_cifar10_batch(nrows=8, ncols=8, show=True)

    # setup tensorboard
    # dts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # summary_dir = os.path.join(TORCH_RUN_SUMMARY_DIR, 'CIFAR10_' + dts)
    # board = Tensorboard(summary_dir)
    board = None

    if 0:
        vis = vis_cifar10_batch(nrows=8, ncols=8)
        board.add_image('CIFAR10_images', vis._im_grid)

    # training params
    num_epochs = 80
    lr = 0.01
    momentum = 0.5

    # init, train, test
    scheduler_mode = 'multistep'
    if scheduler_mode == 'exponential':
        net, criterion, optimizer, scheduler = init_net(lr, momentum, scheduler_mode=scheduler_mode,
                                                        scheduler_params=dict(gamma=0.80))
    if scheduler_mode == 'plateau':
        net, criterion, optimizer, scheduler = init_net(lr, momentum, scheduler_mode=scheduler_mode)
    if scheduler_mode == 'multistep':
        net, criterion, optimizer, scheduler = init_net(lr, momentum, scheduler_mode=scheduler_mode,
                                                        num_epochs=num_epochs)

    train(trainloader, net, criterion, optimizer, num_epochs, board=board, testloader=testloader,
          scheduler_mode=scheduler_mode, scheduler=scheduler)

    eval_classifier_test_acc(testloader, net)




if __name__ == '__main__':
    try_net_on_cifar10()