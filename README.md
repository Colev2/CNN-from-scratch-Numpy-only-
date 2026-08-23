1st run: 
lr = 0.001
optimizer: Adam(b1=0.9, b2=0.999, epsilon=1e-8)
Architecture:
    model = Sequential([
        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Flatten(),

        Dense(neurons=256, rng=rng, initialization="he", distribution="normal"),
        BatchNorm(epsilon=1e-5, momentum=0.9),
        ReLU(),

        Dropout(drop_prob=0.4, rng=rng),

        Dense(neurons=len(np.unique(labels)), rng=rng, initialization="xavier", distribution="normal"),
            ])

    model.build(X_train.shape[1:])
    optimizer = create_optimizer_object(model, optimizer_class, learning_rate)

    # Early Stopping
    early_stopping = EarlyStopping(patience=7, min_delta=1e-3)

    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, min_delta=1e-3, min_lr=1e-5)


1st run results:
3
Choose epochs: 20
Choose weights' initialization for Convolutional Layers (He or Xavier): he
Choose weights' distribution for Convolutional Layers (Uniform or Normal): normal
Choose learning rate: 0.001
Choose optimizer:
1) SGD
2) SGD_momentum
3) Adam
4) AdamW
3
Training epoch 1...
Epoch 1: train_acc: 56.8222% , train_loss: 1.2293 | val_acc: 66.9600% , val_loss: 0.9413 | learning_rate: 0.0010
Training epoch 2...
Epoch 2: train_acc: 70.3933% , train_loss: 0.8500 | val_acc: 70.9200% , val_loss: 0.8149 | learning_rate: 0.0010
Training epoch 3...
Epoch 3: train_acc: 75.2400% , train_loss: 0.7053 | val_acc: 76.1000% , val_loss: 0.6970 | learning_rate: 0.0010
Training epoch 4...
Epoch 4: train_acc: 78.7956% , train_loss: 0.6068 | val_acc: 78.4800% , val_loss: 0.6248 | learning_rate: 0.0010
Training epoch 5...
Epoch 5: train_acc: 82.5289% , train_loss: 0.5069 | val_acc: 79.0800% , val_loss: 0.6163 | learning_rate: 0.0010
Training epoch 6...
Epoch 6: train_acc: 85.1578% , train_loss: 0.4258 | val_acc: 77.9800% , val_loss: 0.6668 | learning_rate: 0.0010
Training epoch 7...
Epoch 7: train_acc: 87.0111% , train_loss: 0.3671 | val_acc: 79.1800% , val_loss: 0.6351 | learning_rate: 0.0010
Training epoch 8...
Epoch 8: train_acc: 89.2111% , train_loss: 0.3078 | val_acc: 80.5200% , val_loss: 0.6230 | learning_rate: 0.0010
Training epoch 9...
Epoch 9: train_acc: 90.4956% , train_loss: 0.2699 | val_acc: 79.3600% , val_loss: 0.6685 | learning_rate: 0.0010
Training epoch 10...
Epoch 10: train_acc: 94.0933% , train_loss: 0.1732 | val_acc: 81.2800% , val_loss: 0.6395 | learning_rate: 0.0005
Training epoch 11...
Epoch 11: train_acc: 95.3889% , train_loss: 0.1351 | val_acc: 80.4200% , val_loss: 0.7021 | learning_rate: 0.0005
Training epoch 12...
Epoch 12: train_acc: 96.0089% , train_loss: 0.1172 | val_acc: 80.2000% , val_loss: 0.7130 | learning_rate: 0.0005
Training epoch 13...
Epoch 13: train_acc: 96.3778% , train_loss: 0.1053 | val_acc: 80.3400% , val_loss: 0.7466 | learning_rate: 0.0005
Best weights are at epoch: 5


2nd run:
lr = 0.0005
optimizer: Adam(b1=0.9, b2=0.999, epsilon=1e-8)
Same Architecture



2nd run results:
Choose epochs: 25
Choose weights' initialization for Convolutional Layers (He or Xavier): he
Choose weights' distribution for Convolutional Layers (Uniform or Normal): normal
Choose learning rate: 0.0005
Choose optimizer:
1) SGD
2) SGD_momentum
3) Adam
4) AdamW
3
Training epoch 1...
Epoch 1: train_acc: 54.2178% , train_loss: 1.3048 | val_acc: 64.7800% , val_loss: 1.0010 | learning_rate: 0.0005
Training epoch 2...
Epoch 2: train_acc: 68.3467% , train_loss: 0.9041 | val_acc: 70.2800% , val_loss: 0.8275 | learning_rate: 0.0005
Training epoch 3...
Epoch 3: train_acc: 73.9467% , train_loss: 0.7443 | val_acc: 72.7400% , val_loss: 0.7730 | learning_rate: 0.0005
Training epoch 4...
Epoch 4: train_acc: 77.4956% , train_loss: 0.6407 | val_acc: 76.5000% , val_loss: 0.6690 | learning_rate: 0.0005
Training epoch 5...
Epoch 5: train_acc: 81.3311% , train_loss: 0.5385 | val_acc: 75.2600% , val_loss: 0.6989 | learning_rate: 0.0005
Training epoch 6...
Epoch 6: train_acc: 84.4089% , train_loss: 0.4501 | val_acc: 75.0600% , val_loss: 0.7521 | learning_rate: 0.0005
Training epoch 7...
Epoch 7: train_acc: 86.3933% , train_loss: 0.3886 | val_acc: 76.4600% , val_loss: 0.7065 | learning_rate: 0.0005
Training epoch 8...
Epoch 8: train_acc: 88.5244% , train_loss: 0.3283 | val_acc: 76.3400% , val_loss: 0.7137 | learning_rate: 0.0005
Training epoch 9...
Epoch 9: train_acc: 92.5200% , train_loss: 0.2208 | val_acc: 77.6200% , val_loss: 0.7044 | learning_rate: 0.0003
Training epoch 10...
Epoch 10: train_acc: 94.1867% , train_loss: 0.1759 | val_acc: 77.7600% , val_loss: 0.7195 | learning_rate: 0.0003
Training epoch 11...
Epoch 11: train_acc: 94.9578% , train_loss: 0.1533 | val_acc: 76.8200% , val_loss: 0.7640 | learning_rate: 0.0003
Training epoch 12...
Epoch 12: train_acc: 95.8200% , train_loss: 0.1293 | val_acc: 77.4800% , val_loss: 0.7955 | learning_rate: 0.0003
Best weights are at epoch: 4





3rd run:
lr = 0.001
optimizer: Adam(same parameter values)
Dropout rate = 0.5
Same architecture
REMOVED:
rng.shuffle(train_indexes) 
rng.shuffle(val_indexes)
FROM TRAIN_TEST_SPLIT SO RNG CHANGES -> WEIGHT INITIALIZATION CHANGES


3rd run results:
Choose epochs: 25
Choose weights' initialization for Convolutional Layers (He or Xavier): he
Choose weights' distribution for Convolutional Layers (Uniform or Normal): normal
Choose learning rate: 0.001
Choose optimizer:
1) SGD
2) SGD_momentum
3) Adam
4) AdamW
3
Training epoch 1...
Epoch 1: train_acc: 54.9756% , train_loss: 1.2897 | val_acc: 67.1600% , val_loss: 0.9160 | learning_rate: 0.0010
Training epoch 2...
Epoch 2: train_acc: 68.3111% , train_loss: 0.8986 | val_acc: 72.1000% , val_loss: 0.7972 | learning_rate: 0.0010
Training epoch 3...
Epoch 3: train_acc: 73.3467% , train_loss: 0.7623 | val_acc: 75.4600% , val_loss: 0.7184 | learning_rate: 0.0010
Training epoch 4...
Epoch 4: train_acc: 77.0156% , train_loss: 0.6587 | val_acc: 77.9800% , val_loss: 0.6253 | learning_rate: 0.0010
Training epoch 5...
Epoch 5: train_acc: 79.8489% , train_loss: 0.5740 | val_acc: 78.8600% , val_loss: 0.6212 | learning_rate: 0.0010
Training epoch 6...
Epoch 6: train_acc: 82.5133% , train_loss: 0.5006 | val_acc: 78.4800% , val_loss: 0.6158 | learning_rate: 0.0010
Training epoch 7...
Epoch 7: train_acc: 84.7844% , train_loss: 0.4321 | val_acc: 79.1400% , val_loss: 0.6318 | learning_rate: 0.0010
Training epoch 8...
Epoch 8: train_acc: 86.4578% , train_loss: 0.3806 | val_acc: 79.5200% , val_loss: 0.6035 | learning_rate: 0.0010
Training epoch 9...
Epoch 9: train_acc: 88.1356% , train_loss: 0.3345 | val_acc: 80.2600% , val_loss: 0.6096 | learning_rate: 0.0010
Training epoch 10...
Epoch 10: train_acc: 89.4711% , train_loss: 0.3026 | val_acc: 80.6200% , val_loss: 0.6075 | learning_rate: 0.0010
Training epoch 11...
Epoch 11: train_acc: 90.5311% , train_loss: 0.2651 | val_acc: 80.0600% , val_loss: 0.6477 | learning_rate: 0.0010
Training epoch 12...
Epoch 12: train_acc: 91.5289% , train_loss: 0.2414 | val_acc: 80.3000% , val_loss: 0.6492 | learning_rate: 0.0010
Training epoch 13...
Epoch 13: train_acc: 94.1422% , train_loss: 0.1657 | val_acc: 81.2000% , val_loss: 0.6448 | learning_rate: 0.0005
Training epoch 14...
Epoch 14: train_acc: 95.4400% , train_loss: 0.1337 | val_acc: 80.8800% , val_loss: 0.6856 | learning_rate: 0.0005
Training epoch 15...
Epoch 15: train_acc: 95.8222% , train_loss: 0.1208 | val_acc: 80.4000% , val_loss: 0.7291 | learning_rate: 0.0005
Training epoch 16...
Epoch 16: train_acc: 96.2533% , train_loss: 0.1088 | val_acc: 81.1200% , val_loss: 0.7499 | learning_rate: 0.0005
Best weights are at epoch: 8




4th run:
lr = 0.001
optimizer = Adam(same param values)
Changed batch norm momentum from 0.9 -> 0.99
   model = Sequential([
        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.99),
        ReLU(),

        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.99),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.99),
        ReLU(),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-5, momentum=0.99),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Flatten(),

        Dense(neurons=256, rng=rng, initialization="he", distribution="normal"),
        BatchNorm(epsilon=1e-5, momentum=0.99),
        ReLU(),

        Dropout(drop_prob=0.5, rng=rng),

        Dense(neurons=len(np.unique(labels)), rng=rng, initialization="xavier", distribution="normal"),
            ])

    model.build(X_train.shape[1:])
    optimizer = create_optimizer_object(model, optimizer_class, learning_rate, weight_decay=weight_decay)

    # Early Stopping
    early_stopping = EarlyStopping(patience=7, min_delta=1e-3)

    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, min_delta=1e-3, min_lr=1e-5)


4th run results:

Choose epochs: 25
Choose weights' initialization for Convolutional Layers (He or Xavier): he
Choose weights' distribution for Convolutional Layers (Uniform or Normal): normal
Choose learning rate: 0.001
Choose optimizer:
1) SGD
2) SGD_momentum
3) Adam
4) AdamW
3
Training epoch 1...
Epoch 1: train_acc: 54.9756% , train_loss: 1.2897 | val_acc: 61.7200% , val_loss: 1.0835 | learning_rate: 0.0010
Training epoch 2...
Epoch 2: train_acc: 68.3111% , train_loss: 0.8986 | val_acc: 72.1000% , val_loss: 0.7989 | learning_rate: 0.0010
Training epoch 3...
Epoch 3: train_acc: 73.3467% , train_loss: 0.7623 | val_acc: 74.4400% , val_loss: 0.7451 | learning_rate: 0.0010
Training epoch 4...
Epoch 4: train_acc: 77.0156% , train_loss: 0.6587 | val_acc: 74.7200% , val_loss: 0.7265 | learning_rate: 0.0010
Training epoch 5...
Epoch 5: train_acc: 79.8489% , train_loss: 0.5740 | val_acc: 77.6000% , val_loss: 0.6180 | learning_rate: 0.0010
Training epoch 6...
Epoch 6: train_acc: 82.5133% , train_loss: 0.5006 | val_acc: 76.3200% , val_loss: 0.6845 | learning_rate: 0.0010
Training epoch 7...
Epoch 7: train_acc: 84.7844% , train_loss: 0.4321 | val_acc: 76.1600% , val_loss: 0.7254 | learning_rate: 0.0010
Training epoch 8...
Epoch 8: train_acc: 86.4578% , train_loss: 0.3806 | val_acc: 77.1000% , val_loss: 0.6862 | learning_rate: 0.0010
Training epoch 9...
Epoch 9: train_acc: 88.1356% , train_loss: 0.3345 | val_acc: 78.5000% , val_loss: 0.7000 | learning_rate: 0.0010
Training epoch 10...
Epoch 10: train_acc: 91.8689% , train_loss: 0.2356 | val_acc: 81.4400% , val_loss: 0.6026 | learning_rate: 0.0005
Training epoch 11...
Epoch 11: train_acc: 93.2533% , train_loss: 0.1928 | val_acc: 80.4800% , val_loss: 0.6553 | learning_rate: 0.0005
Training epoch 12...
Epoch 12: train_acc: 94.1600% , train_loss: 0.1676 | val_acc: 80.2800% , val_loss: 0.6880 | learning_rate: 0.0005
Training epoch 13...
Epoch 13: train_acc: 94.7822% , train_loss: 0.1515 | val_acc: 79.9600% , val_loss: 0.7223 | learning_rate: 0.0005
Training epoch 14...
Epoch 14: train_acc: 95.3133% , train_loss: 0.1373 | val_acc: 80.7000% , val_loss: 0.7117 | learning_rate: 0.0005
Training epoch 15...
Epoch 15: train_acc: 96.5489% , train_loss: 0.1029 | val_acc: 81.0000% , val_loss: 0.6933 | learning_rate: 0.0003
Training epoch 16...
Epoch 16: train_acc: 97.0422% , train_loss: 0.0881 | val_acc: 81.0600% , val_loss: 0.7339 | learning_rate: 0.0003
Training epoch 17...
Epoch 17: train_acc: 97.1933% , train_loss: 0.0824 | val_acc: 80.9400% , val_loss: 0.7334 | learning_rate: 0.0003
Training epoch 18...
Epoch 18: train_acc: 97.5622% , train_loss: 0.0744 | val_acc: 80.7800% , val_loss: 0.7587 | learning_rate: 0.0003
Best weights are at epoch: 10


5th run:
Everything same as 4th but with AdamW(weight_decay=1e-4)


5th run results:
Choose epochs: 25
Choose weights' initialization for Convolutional Layers (He or Xavier): he
Choose weights' distribution for Convolutional Layers (Uniform or Normal): normal
Choose learning rate: 0.001
Choose optimizer:
1) SGD
2) SGD_momentum
3) Adam
4) AdamW
4
Give weight decay: 0.0001
Training epoch 1...
Epoch 1: train_acc: 54.7711% , train_loss: 1.2887 | val_acc: 64.5200% , val_loss: 1.0009 | learning_rate: 0.0010
Training epoch 2...
Epoch 2: train_acc: 68.1511% , train_loss: 0.8966 | val_acc: 71.2000% , val_loss: 0.8240 | learning_rate: 0.0010
Training epoch 3...
Epoch 3: train_acc: 73.4956% , train_loss: 0.7620 | val_acc: 73.6600% , val_loss: 0.7578 | learning_rate: 0.0010
Training epoch 4...
Epoch 4: train_acc: 76.8222% , train_loss: 0.6574 | val_acc: 72.8400% , val_loss: 0.7859 | learning_rate: 0.0010
Training epoch 5...
Epoch 5: train_acc: 80.1911% , train_loss: 0.5684 | val_acc: 77.0200% , val_loss: 0.6603 | learning_rate: 0.0010
Training epoch 6...
Epoch 6: train_acc: 82.7356% , train_loss: 0.4959 | val_acc: 77.2200% , val_loss: 0.6922 | learning_rate: 0.0010
Training epoch 7...
Epoch 7: train_acc: 84.7867% , train_loss: 0.4324 | val_acc: 75.9000% , val_loss: 0.7566 | learning_rate: 0.0010
Training epoch 8...
Epoch 8: train_acc: 86.7378% , train_loss: 0.3772 | val_acc: 77.4200% , val_loss: 0.7344 | learning_rate: 0.0010
Training epoch 9...
Epoch 9: train_acc: 88.4689% , train_loss: 0.3324 | val_acc: 77.2800% , val_loss: 0.7428 | learning_rate: 0.0010
Training epoch 10...
Epoch 10: train_acc: 91.8711% , train_loss: 0.2323 | val_acc: 80.4600% , val_loss: 0.6087 | learning_rate: 0.0005
Training epoch 11...
Epoch 11: train_acc: 93.4489% , train_loss: 0.1890 | val_acc: 79.8800% , val_loss: 0.6649 | learning_rate: 0.0005
Training epoch 12...
Epoch 12: train_acc: 94.3289% , train_loss: 0.1659 | val_acc: 80.5000% , val_loss: 0.6606 | learning_rate: 0.0005
Training epoch 13...
Epoch 13: train_acc: 94.8844% , train_loss: 0.1477 | val_acc: 79.5400% , val_loss: 0.7323 | learning_rate: 0.0005
Training epoch 14...
Epoch 14: train_acc: 95.2511% , train_loss: 0.1350 | val_acc: 81.1000% , val_loss: 0.7009 | learning_rate: 0.0005
Training epoch 15...
Epoch 15: train_acc: 96.7422% , train_loss: 0.0979 | val_acc: 81.7800% , val_loss: 0.7136 | learning_rate: 0.0003
Training epoch 16...
Epoch 16: train_acc: 97.0400% , train_loss: 0.0898 | val_acc: 81.2400% , val_loss: 0.7255 | learning_rate: 0.0003
Training epoch 17...
Epoch 17: train_acc: 97.2378% , train_loss: 0.0797 | val_acc: 81.2400% , val_loss: 0.7228 | learning_rate: 0.0003
Training epoch 18...
Epoch 18: train_acc: 97.5289% , train_loss: 0.0741 | val_acc: 80.9000% , val_loss: 0.7930 | learning_rate: 0.0003
Best weights are at epoch: 10



6th run:
Changed Dropout layer position: Now it is after Flatten Layer
