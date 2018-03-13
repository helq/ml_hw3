import struct
import gzip
import numpy as np

def load_images(dataset = 'training'):
    if dataset not in ['training', 'testing']:
        raise Exception("Selected dataset name is not a valid dataset name")

    path = '5x46789x9x18x6x2x96x96-training' if dataset == 'training' else '5x01235x9x18x6x2x96x96-testing'

    with gzip.open('dataset/smallnorb-{}-dat.mat.gz'.format(path), 'rb') as f:
        header, ndim = struct.unpack('<ii', f.read(8))
        assert (0x1E3D4C55, 4) == (header, ndim)
        ints_to_read = ndim if ndim>2 else 3
        dims = f.read(ints_to_read*4)
        assert (24300, 2, 96, 96) == struct.unpack('<'+'i'*ints_to_read, dims)

        set_ = np.zeros( (24300, 2, 96, 96), dtype='B' )
        for i in range(int(24300/100)):
            # loading in chunks of 100 images
            #print(i)
            set_[100*i:100*(i+1)] = np.fromstring( f.read(100*2*96*96), dtype='<B' ).reshape(100,2,96,96)

    return set_

def load_labels(dataset = 'training'):
    if dataset not in ['training', 'testing']:
        raise Exception("Selected dataset name is not a valid dataset name")

    path = '5x46789x9x18x6x2x96x96-training' if dataset == 'training' else '5x01235x9x18x6x2x96x96-testing'

    with gzip.open('dataset/smallnorb-{}-cat.mat.gz'.format(path), 'rb') as f:
        header, ndim = struct.unpack('<ii', f.read(8))
        assert (0x1E3D4C54, 1) == (header, ndim)
        ints_to_read = ndim if ndim>2 else 3
        dims = f.read(ints_to_read*4)
        assert (24300, 1, 1) == struct.unpack('<'+'i'*ints_to_read, dims)

        set_ = np.fromstring( f.read(), dtype='<i' ).reshape( (24300,) )

    return set_

def load_set(dataset):
    images = load_images(dataset).transpose( (0,3,1,2) ).reshape( (24300, 2*96*96) ).astype( np.float32 ) / 256
    labels = load_labels(dataset).astype( np.uint8 )
    return images, labels

def show_image_pair(imgs):
    import matplotlib.pyplot as plt

    fig1 = plt.figure()
    plt.imshow(imgs[0], vmin=0, vmax=255, cmap='gray')

    fig2 = plt.figure()
    plt.imshow(imgs[1], vmin=0, vmax=255, cmap='gray')

    fig1.show()
    fig2.show()

if __name__ == '__main__':
    training_imgs = load_images('testing')
    training_labels = load_labels('testing')

    img_id = 6452
    print( "Displaying image {} with label {} (from testing)".format(img_id, training_labels[img_id]) )
    show_image_pair(training_imgs[img_id])
    #plt.show()
