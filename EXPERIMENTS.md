CIFAR-10 EXPERIMENTS

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


4th run results:

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


6th run results:

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
Epoch 1: train_acc: 54.7356% , train_loss: 1.2664 | val_acc: 62.4800% , val_loss: 1.0724 | learning_rate: 0.0010
Training epoch 2...
Epoch 2: train_acc: 68.0244% , train_loss: 0.9027 | val_acc: 73.5200% , val_loss: 0.7650 | learning_rate: 0.0010
Training epoch 3...
Epoch 3: train_acc: 73.0333% , train_loss: 0.7686 | val_acc: 74.2400% , val_loss: 0.7236 | learning_rate: 0.0010
Training epoch 4...
Epoch 4: train_acc: 76.0667% , train_loss: 0.6785 | val_acc: 74.1000% , val_loss: 0.7209 | learning_rate: 0.0010
Training epoch 5...
Epoch 5: train_acc: 78.4867% , train_loss: 0.6112 | val_acc: 77.4800% , val_loss: 0.6419 | learning_rate: 0.0010
Training epoch 6...
Epoch 6: train_acc: 80.6667% , train_loss: 0.5476 | val_acc: 78.9200% , val_loss: 0.6179 | learning_rate: 0.0010
Training epoch 7...
Epoch 7: train_acc: 82.6444% , train_loss: 0.4986 | val_acc: 75.8000% , val_loss: 0.6855 | learning_rate: 0.0010
Training epoch 8...
Epoch 8: train_acc: 84.0889% , train_loss: 0.4564 | val_acc: 80.5200% , val_loss: 0.5416 | learning_rate: 0.0010
Training epoch 9...
Epoch 9: train_acc: 85.5244% , train_loss: 0.4157 | val_acc: 81.2200% , val_loss: 0.5466 | learning_rate: 0.0010
Training epoch 10...
Epoch 10: train_acc: 86.4600% , train_loss: 0.3855 | val_acc: 80.1200% , val_loss: 0.5916 | learning_rate: 0.0010
Training epoch 11...
Epoch 11: train_acc: 87.6000% , train_loss: 0.3508 | val_acc: 79.7400% , val_loss: 0.6041 | learning_rate: 0.0010
Training epoch 12...
Epoch 12: train_acc: 88.6578% , train_loss: 0.3229 | val_acc: 81.6400% , val_loss: 0.5595 | learning_rate: 0.0010
Training epoch 13...
Epoch 13: train_acc: 91.2733% , train_loss: 0.2519 | val_acc: 83.6000% , val_loss: 0.4767 | learning_rate: 0.0005
Training epoch 14...
Epoch 14: train_acc: 92.0178% , train_loss: 0.2281 | val_acc: 84.5400% , val_loss: 0.4633 | learning_rate: 0.0005
Training epoch 15...
Epoch 15: train_acc: 92.7067% , train_loss: 0.2094 | val_acc: 84.5200% , val_loss: 0.4899 | learning_rate: 0.0005
Training epoch 16...
Epoch 16: train_acc: 93.0422% , train_loss: 0.1987 | val_acc: 84.4000% , val_loss: 0.4917 | learning_rate: 0.0005
Training epoch 17...
Epoch 17: train_acc: 93.6444% , train_loss: 0.1823 | val_acc: 83.5800% , val_loss: 0.5164 | learning_rate: 0.0005
Training epoch 18...
Epoch 18: train_acc: 94.0022% , train_loss: 0.1738 | val_acc: 83.6600% , val_loss: 0.5163 | learning_rate: 0.0005
Training epoch 19...
Epoch 19: train_acc: 95.0378% , train_loss: 0.1465 | val_acc: 84.6400% , val_loss: 0.4848 | learning_rate: 0.0003
Training epoch 20...
Epoch 20: train_acc: 95.1889% , train_loss: 0.1407 | val_acc: 85.0000% , val_loss: 0.4879 | learning_rate: 0.0003
Training epoch 21...
Epoch 21: train_acc: 95.7822% , train_loss: 0.1270 | val_acc: 84.7200% , val_loss: 0.4965 | learning_rate: 0.0003
Training epoch 22...
Epoch 22: train_acc: 95.8511% , train_loss: 0.1230 | val_acc: 84.8600% , val_loss: 0.4861 | learning_rate: 0.0003
Best weights are at epoch: 14


7th run: 

Changed BatchNorm epsilon from e = 1e-5 to e = 1e-3


7th run results:

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
Epoch 1: train_acc: 54.5533% , train_loss: 1.2802 | val_acc: 59.5000% , val_loss: 1.1937 | learning_rate: 0.0010
Training epoch 2...
Epoch 2: train_acc: 67.9378% , train_loss: 0.9133 | val_acc: 69.9000% , val_loss: 0.8592 | learning_rate: 0.0010
Training epoch 3...
Epoch 3: train_acc: 72.8422% , train_loss: 0.7760 | val_acc: 72.5200% , val_loss: 0.7936 | learning_rate: 0.0010
Training epoch 4...
Epoch 4: train_acc: 76.0178% , train_loss: 0.6829 | val_acc: 73.2200% , val_loss: 0.7518 | learning_rate: 0.0010
Training epoch 5...
Epoch 5: train_acc: 78.4844% , train_loss: 0.6142 | val_acc: 77.8800% , val_loss: 0.6183 | learning_rate: 0.0010
Training epoch 6...
Epoch 6: train_acc: 80.7111% , train_loss: 0.5491 | val_acc: 78.8800% , val_loss: 0.6094 | learning_rate: 0.0010
Training epoch 7...
Epoch 7: train_acc: 82.2711% , train_loss: 0.5022 | val_acc: 77.0000% , val_loss: 0.6523 | learning_rate: 0.0010
Training epoch 8...
Epoch 8: train_acc: 83.7556% , train_loss: 0.4613 | val_acc: 77.9200% , val_loss: 0.6607 | learning_rate: 0.0010
Training epoch 9...
Epoch 9: train_acc: 85.3911% , train_loss: 0.4187 | val_acc: 81.8200% , val_loss: 0.5336 | learning_rate: 0.0010
Training epoch 10...
Epoch 10: train_acc: 86.4356% , train_loss: 0.3855 | val_acc: 82.4000% , val_loss: 0.5223 | learning_rate: 0.0010
Training epoch 11...
Epoch 11: train_acc: 87.7133% , train_loss: 0.3492 | val_acc: 80.0000% , val_loss: 0.5907 | learning_rate: 0.0010
Training epoch 12...
Epoch 12: train_acc: 88.3622% , train_loss: 0.3287 | val_acc: 82.0600% , val_loss: 0.5555 | learning_rate: 0.0010
Training epoch 13...
Epoch 13: train_acc: 89.4644% , train_loss: 0.3013 | val_acc: 82.4800% , val_loss: 0.5205 | learning_rate: 0.0010
Training epoch 14...
Epoch 14: train_acc: 90.2489% , train_loss: 0.2770 | val_acc: 81.0200% , val_loss: 0.6207 | learning_rate: 0.0010
Training epoch 15...
Epoch 15: train_acc: 90.6378% , train_loss: 0.2637 | val_acc: 83.3400% , val_loss: 0.5238 | learning_rate: 0.0010
Training epoch 16...
Epoch 16: train_acc: 91.1711% , train_loss: 0.2476 | val_acc: 81.1800% , val_loss: 0.6034 | learning_rate: 0.0010
Training epoch 17...
Epoch 17: train_acc: 91.8356% , train_loss: 0.2356 | val_acc: 81.6200% , val_loss: 0.5893 | learning_rate: 0.0010
Training epoch 18...
Epoch 18: train_acc: 93.7578% , train_loss: 0.1817 | val_acc: 84.1600% , val_loss: 0.5151 | learning_rate: 0.0005
Training epoch 19...
Epoch 19: train_acc: 94.6156% , train_loss: 0.1598 | val_acc: 84.0800% , val_loss: 0.5308 | learning_rate: 0.0005
Training epoch 20...
Epoch 20: train_acc: 94.7156% , train_loss: 0.1526 | val_acc: 83.8200% , val_loss: 0.5250 | learning_rate: 0.0005
Training epoch 21...
Epoch 21: train_acc: 95.1178% , train_loss: 0.1416 | val_acc: 84.2400% , val_loss: 0.5236 | learning_rate: 0.0005
Training epoch 22...
Epoch 22: train_acc: 95.1111% , train_loss: 0.1393 | val_acc: 83.8600% , val_loss: 0.5301 | learning_rate: 0.0005
Training epoch 23...
Epoch 23: train_acc: 95.9733% , train_loss: 0.1184 | val_acc: 85.0800% , val_loss: 0.5118 | learning_rate: 0.0003
Training epoch 24...
Epoch 24: train_acc: 96.1378% , train_loss: 0.1107 | val_acc: 84.8400% , val_loss: 0.5188 | learning_rate: 0.0003
Training epoch 25...
Epoch 25: train_acc: 96.6178% , train_loss: 0.1022 | val_acc: 85.3600% , val_loss: 0.5142 | learning_rate: 0.0003
Best weights are at epoch: 23



8th run:
Added Data augmentation:
    model = Sequential([
        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        Conv2D(filters=32, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        Conv2D(filters=64, filter_shape=(3,3), padding=1, stride=1, rng=rng, initialization=initialization, distribution=distribution),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        MaxPooling2D(pool_size=(2,2), stride=2),

        Flatten(),
        Dropout(drop_prob=0.5, rng=rng),
        
        Dense(neurons=256, rng=rng, initialization="he", distribution="normal"),
        BatchNorm(epsilon=1e-3, momentum=0.99),
        ReLU(),

        Dense(neurons=len(np.unique(labels)), rng=rng, initialization="xavier", distribution="normal"),
        ])

    model.build(X_train.shape[1:])
    optimizer = create_optimizer_object(model, optimizer_class, learning_rate, weight_decay=weight_decay)

    # Early Stopping
    early_stopping = EarlyStopping(patience=7, min_delta=1e-3)



8th run results:

Choose epochs: 30
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
Epoch 1: train_acc: 47.3933% , train_loss: 1.4575 | val_acc: 59.9200% , val_loss: 1.1413 | learning_rate: 0.0010
Training epoch 2...
Epoch 2: train_acc: 60.9467% , train_loss: 1.0926 | val_acc: 67.7200% , val_loss: 0.8766 | learning_rate: 0.0010
Training epoch 3...
Epoch 3: train_acc: 66.0156% , train_loss: 0.9605 | val_acc: 73.8000% , val_loss: 0.7439 | learning_rate: 0.0010
Training epoch 4...
Epoch 4: train_acc: 69.5244% , train_loss: 0.8709 | val_acc: 72.7000% , val_loss: 0.7854 | learning_rate: 0.0010
Training epoch 5...
Epoch 5: train_acc: 71.0422% , train_loss: 0.8184 | val_acc: 74.2400% , val_loss: 0.7443 | learning_rate: 0.0010
Training epoch 6...
Epoch 6: train_acc: 72.6000% , train_loss: 0.7757 | val_acc: 77.4000% , val_loss: 0.6290 | learning_rate: 0.0010
Training epoch 7...
Epoch 7: train_acc: 73.7622% , train_loss: 0.7447 | val_acc: 76.0000% , val_loss: 0.6998 | learning_rate: 0.0010
Training epoch 8...
Epoch 8: train_acc: 74.9289% , train_loss: 0.7080 | val_acc: 76.6600% , val_loss: 0.6664 | learning_rate: 0.0010
Training epoch 9...
Epoch 9: train_acc: 75.9200% , train_loss: 0.6839 | val_acc: 78.2200% , val_loss: 0.6349 | learning_rate: 0.0010
Training epoch 10...
Epoch 10: train_acc: 76.8267% , train_loss: 0.6635 | val_acc: 80.3000% , val_loss: 0.5776 | learning_rate: 0.0010
Training epoch 11...
Epoch 11: train_acc: 77.6244% , train_loss: 0.6417 | val_acc: 80.2600% , val_loss: 0.5609 | learning_rate: 0.0010
Training epoch 12...
Epoch 12: train_acc: 78.3911% , train_loss: 0.6194 | val_acc: 81.2200% , val_loss: 0.5323 | learning_rate: 0.0010
Training epoch 13...
Epoch 13: train_acc: 78.4933% , train_loss: 0.6149 | val_acc: 80.4800% , val_loss: 0.5677 | learning_rate: 0.0010
Training epoch 14...
Epoch 14: train_acc: 79.1222% , train_loss: 0.5986 | val_acc: 79.5000% , val_loss: 0.6238 | learning_rate: 0.0010
Training epoch 15...
Epoch 15: train_acc: 79.7733% , train_loss: 0.5801 | val_acc: 83.4400% , val_loss: 0.4794 | learning_rate: 0.0010
Training epoch 16...
Epoch 16: train_acc: 80.1356% , train_loss: 0.5695 | val_acc: 82.8600% , val_loss: 0.4848 | learning_rate: 0.0010
Training epoch 17...
Epoch 17: train_acc: 80.4311% , train_loss: 0.5623 | val_acc: 82.5200% , val_loss: 0.5042 | learning_rate: 0.0010
Training epoch 18...
Epoch 18: train_acc: 80.8800% , train_loss: 0.5513 | val_acc: 82.8400% , val_loss: 0.5023 | learning_rate: 0.0010
Training epoch 19...
Epoch 19: train_acc: 81.2133% , train_loss: 0.5409 | val_acc: 82.0200% , val_loss: 0.5179 | learning_rate: 0.0010
Training epoch 20...
Epoch 20: train_acc: 82.4644% , train_loss: 0.5042 | val_acc: 86.0000% , val_loss: 0.4055 | learning_rate: 0.0005
Training epoch 21...
Epoch 21: train_acc: 83.2756% , train_loss: 0.4878 | val_acc: 86.3000% , val_loss: 0.3942 | learning_rate: 0.0005
Training epoch 22...
Epoch 22: train_acc: 83.2333% , train_loss: 0.4818 | val_acc: 86.7000% , val_loss: 0.3974 | learning_rate: 0.0005
Training epoch 23...
Epoch 23: train_acc: 83.4667% , train_loss: 0.4731 | val_acc: 86.6200% , val_loss: 0.3932 | learning_rate: 0.0005
Training epoch 24...
Epoch 24: train_acc: 83.8044% , train_loss: 0.4685 | val_acc: 86.7200% , val_loss: 0.3833 | learning_rate: 0.0005
Training epoch 25...
Epoch 25: train_acc: 83.9711% , train_loss: 0.4651 | val_acc: 86.5800% , val_loss: 0.3931 | learning_rate: 0.0005
Training epoch 26...
Epoch 26: train_acc: 83.9600% , train_loss: 0.4602 | val_acc: 86.3200% , val_loss: 0.3935 | learning_rate: 0.0005
Training epoch 27...
Epoch 27: train_acc: 84.3244% , train_loss: 0.4531 | val_acc: 86.7000% , val_loss: 0.3969 | learning_rate: 0.0005
Training epoch 28...
Epoch 28: train_acc: 84.3333% , train_loss: 0.4462 | val_acc: 85.8800% , val_loss: 0.4116 | learning_rate: 0.0005
Training epoch 29...
Epoch 29: train_acc: 84.7111% , train_loss: 0.4326 | val_acc: 87.5200% , val_loss: 0.3634 | learning_rate: 0.0003
Training epoch 30...
Epoch 30: train_acc: 85.1311% , train_loss: 0.4258 | val_acc: 87.4200% , val_loss: 0.3657 | learning_rate: 0.0003
Best weights are at epoch: 29



9th run:

Everything same as 8th run, but on CIFAR-100


9th run results:

Selected dataset: CIFAR-100

Train model? (yes/no): y
Choose epochs: 30
Choose weights' initialization for Convolutional Layers (He or Xavier): he
Choose weights' distribution for Convolutional Layers (Uniform or Normal): normal
Choose learning rate: 0.001
Choose optimizer:
1) SGD
2) SGD_momentum
3) Adam
4) AdamW
3
Use data augmentation? (yes/no): y

Training epoch 1...
Epoch 1: train_acc: 16.2422% , train_loss: 3.5527 | val_acc: 28.4200% , val_loss: 2.8264 | learning_rate: 0.00100

Training epoch 2...
Epoch 2: train_acc: 28.2222% , train_loss: 2.8584 | val_acc: 37.6400% , val_loss: 2.4035 | learning_rate: 0.00100

Training epoch 3...
Epoch 3: train_acc: 33.2778% , train_loss: 2.5962 | val_acc: 39.4000% , val_loss: 2.3052 | learning_rate: 0.00100

Training epoch 4...
Epoch 4: train_acc: 36.7289% , train_loss: 2.4357 | val_acc: 44.6000% , val_loss: 2.0861 | learning_rate: 0.00100

Training epoch 5...
Epoch 5: train_acc: 38.9244% , train_loss: 2.3256 | val_acc: 45.7600% , val_loss: 2.0066 | learning_rate: 0.00100

Training epoch 6...
Epoch 6: train_acc: 41.3644% , train_loss: 2.2204 | val_acc: 47.6600% , val_loss: 1.9357 | learning_rate: 0.00100

Training epoch 7...
Epoch 7: train_acc: 43.1378% , train_loss: 2.1459 | val_acc: 47.3200% , val_loss: 1.9216 | learning_rate: 0.00100

Training epoch 8...
Epoch 8: train_acc: 44.5444% , train_loss: 2.0754 | val_acc: 49.2800% , val_loss: 1.8819 | learning_rate: 0.00100

Training epoch 9...
Epoch 9: train_acc: 45.1956% , train_loss: 2.0346 | val_acc: 49.3200% , val_loss: 1.8769 | learning_rate: 0.00100

Training epoch 10...
Epoch 10: train_acc: 46.7378% , train_loss: 1.9748 | val_acc: 44.3400% , val_loss: 2.1673 | learning_rate: 0.00100

Training epoch 11...
Epoch 11: train_acc: 47.3778% , train_loss: 1.9358 | val_acc: 51.0000% , val_loss: 1.8045 | learning_rate: 0.00100

Training epoch 12...
Epoch 12: train_acc: 48.5822% , train_loss: 1.8960 | val_acc: 50.2400% , val_loss: 1.8666 | learning_rate: 0.00100

Training epoch 13...
Epoch 13: train_acc: 49.0111% , train_loss: 1.8678 | val_acc: 52.0600% , val_loss: 1.7916 | learning_rate: 0.00100

Training epoch 14...
Epoch 14: train_acc: 50.0022% , train_loss: 1.8311 | val_acc: 51.6400% , val_loss: 1.7844 | learning_rate: 0.00100

Training epoch 15...
Epoch 15: train_acc: 50.4067% , train_loss: 1.8087 | val_acc: 54.2200% , val_loss: 1.6873 | learning_rate: 0.00100

Training epoch 16...
Epoch 16: train_acc: 51.1333% , train_loss: 1.7801 | val_acc: 54.1000% , val_loss: 1.7098 | learning_rate: 0.00100

Training epoch 17...
Epoch 17: train_acc: 51.9311% , train_loss: 1.7585 | val_acc: 52.0600% , val_loss: 1.7850 | learning_rate: 0.00100

Training epoch 18...
Epoch 18: train_acc: 52.0933% , train_loss: 1.7383 | val_acc: 54.3800% , val_loss: 1.7016 | learning_rate: 0.00100

Training epoch 19...
Epoch 19: train_acc: 52.5311% , train_loss: 1.7235 | val_acc: 54.0600% , val_loss: 1.6750 | learning_rate: 0.00100

Training epoch 20...
Epoch 20: train_acc: 53.1778% , train_loss: 1.6925 | val_acc: 53.5200% , val_loss: 1.7153 | learning_rate: 0.00100

Training epoch 21...
Epoch 21: train_acc: 53.5600% , train_loss: 1.6815 | val_acc: 55.4800% , val_loss: 1.6454 | learning_rate: 0.00100

Training epoch 22...
Epoch 22: train_acc: 53.9200% , train_loss: 1.6528 | val_acc: 55.2200% , val_loss: 1.6609 | learning_rate: 0.00100

Training epoch 23...
Epoch 23: train_acc: 54.1222% , train_loss: 1.6473 | val_acc: 55.7200% , val_loss: 1.6406 | learning_rate: 0.00100

Training epoch 24...
Epoch 24: train_acc: 54.3867% , train_loss: 1.6371 | val_acc: 54.8800% , val_loss: 1.6764 | learning_rate: 0.00100

Training epoch 25...
Epoch 25: train_acc: 55.1644% , train_loss: 1.6224 | val_acc: 56.4000% , val_loss: 1.6471 | learning_rate: 0.00100

Training epoch 26...
Epoch 26: train_acc: 55.0511% , train_loss: 1.6089 | val_acc: 55.1000% , val_loss: 1.6738 | learning_rate: 0.00100

Training epoch 27...
Epoch 27: train_acc: 55.3867% , train_loss: 1.5989 | val_acc: 56.0400% , val_loss: 1.5993 | learning_rate: 0.00100

Training epoch 28...
Epoch 28: train_acc: 55.7978% , train_loss: 1.5793 | val_acc: 55.1400% , val_loss: 1.6457 | learning_rate: 0.00100

Training epoch 29...
Epoch 29: train_acc: 56.1800% , train_loss: 1.5699 | val_acc: 54.2800% , val_loss: 1.7157 | learning_rate: 0.00100

Training epoch 30...
Epoch 30: train_acc: 56.3244% , train_loss: 1.5538 | val_acc: 57.4000% , val_loss: 1.5818 | learning_rate: 0.00100

Training finished
Restored Best weights from epoch: 30



10th run: 

Changed Architecture for CIFAR-100:
Added 2 convolutional layers before the flatten layer, with 128 filters.


10th run results:

Training epoch 1...
Epoch 1: train_acc: 14.1067% , train_loss: 3.6662 | val_acc: 19.2800% , val_loss: 3.4311 | learning_rate: 0.00100

Training epoch 2...
Epoch 2: train_acc: 26.3156% , train_loss: 2.9317 | val_acc: 31.3000% , val_loss: 2.7324 | learning_rate: 0.00100

Training epoch 3...
Epoch 3: train_acc: 33.2844% , train_loss: 2.5705 | val_acc: 34.9400% , val_loss: 2.5383 | learning_rate: 0.00100

Training epoch 4...
Epoch 4: train_acc: 38.3756% , train_loss: 2.3464 | val_acc: 42.4000% , val_loss: 2.1557 | learning_rate: 0.00100

Training epoch 5...
Epoch 5: train_acc: 41.2844% , train_loss: 2.1980 | val_acc: 43.6800% , val_loss: 2.1148 | learning_rate: 0.00100

Training epoch 6...
Epoch 6: train_acc: 44.3622% , train_loss: 2.0642 | val_acc: 46.3600% , val_loss: 1.9906 | learning_rate: 0.00100

Training epoch 7...
Epoch 7: train_acc: 46.2733% , train_loss: 1.9755 | val_acc: 45.5000% , val_loss: 2.0362 | learning_rate: 0.00100

Training epoch 8...
Epoch 8: train_acc: 48.0578% , train_loss: 1.8964 | val_acc: 48.6800% , val_loss: 1.9507 | learning_rate: 0.00100

Training epoch 9...
Epoch 9: train_acc: 49.6422% , train_loss: 1.8260 | val_acc: 49.5200% , val_loss: 1.8738 | learning_rate: 0.00100

Training epoch 10...
Epoch 10: train_acc: 51.0089% , train_loss: 1.7733 | val_acc: 53.2600% , val_loss: 1.7191 | learning_rate: 0.00100

Training epoch 11...
Epoch 11: train_acc: 52.1311% , train_loss: 1.7112 | val_acc: 53.0800% , val_loss: 1.6885 | learning_rate: 0.00100

Training epoch 12...
Epoch 12: train_acc: 53.0711% , train_loss: 1.6739 | val_acc: 54.0000% , val_loss: 1.6646 | learning_rate: 0.00100

Training epoch 13...
Epoch 13: train_acc: 54.3022% , train_loss: 1.6308 | val_acc: 54.8400% , val_loss: 1.6303 | learning_rate: 0.00100

Training epoch 14...
Epoch 14: train_acc: 55.2156% , train_loss: 1.6022 | val_acc: 54.0200% , val_loss: 1.6829 | learning_rate: 0.00100

Training epoch 15...
Epoch 15: train_acc: 55.9400% , train_loss: 1.5654 | val_acc: 55.7800% , val_loss: 1.5995 | learning_rate: 0.00100

Training epoch 16...
Epoch 16: train_acc: 56.4067% , train_loss: 1.5430 | val_acc: 56.7200% , val_loss: 1.5473 | learning_rate: 0.00100

Training epoch 17...
Epoch 17: train_acc: 56.6511% , train_loss: 1.5191 | val_acc: 54.4800% , val_loss: 1.6595 | learning_rate: 0.00100

Training epoch 18...
Epoch 18: train_acc: 57.5978% , train_loss: 1.4833 | val_acc: 56.2600% , val_loss: 1.5753 | learning_rate: 0.00100

Training epoch 19...
Epoch 19: train_acc: 58.5356% , train_loss: 1.4619 | val_acc: 57.4800% , val_loss: 1.5012 | learning_rate: 0.00100

Training epoch 20...
Epoch 20: train_acc: 58.9178% , train_loss: 1.4375 | val_acc: 58.4200% , val_loss: 1.5081 | learning_rate: 0.00100

Training epoch 21...
Epoch 21: train_acc: 59.3111% , train_loss: 1.4227 | val_acc: 57.8600% , val_loss: 1.5345 | learning_rate: 0.00100

Training epoch 22...
Epoch 22: train_acc: 60.0778% , train_loss: 1.3924 | val_acc: 59.3600% , val_loss: 1.4794 | learning_rate: 0.00100

Training epoch 23...
Epoch 23: train_acc: 60.6356% , train_loss: 1.3748 | val_acc: 57.7600% , val_loss: 1.5346 | learning_rate: 0.00100

Training epoch 24...
Epoch 24: train_acc: 60.8556% , train_loss: 1.3584 | val_acc: 58.0000% , val_loss: 1.5149 | learning_rate: 0.00100

Training epoch 25...
Epoch 25: train_acc: 61.6622% , train_loss: 1.3380 | val_acc: 57.3000% , val_loss: 1.5410 | learning_rate: 0.00100

Training epoch 26...
Epoch 26: train_acc: 61.8022% , train_loss: 1.3204 | val_acc: 58.3200% , val_loss: 1.5089 | learning_rate: 0.00100

Training epoch 27...
Epoch 27: train_acc: 64.2289% , train_loss: 1.2300 | val_acc: 61.9800% , val_loss: 1.3641 | learning_rate: 0.00050

Training epoch 28...
Epoch 28: train_acc: 64.7689% , train_loss: 1.2016 | val_acc: 61.3000% , val_loss: 1.4004 | learning_rate: 0.00050

Training epoch 29...
Epoch 29: train_acc: 65.3956% , train_loss: 1.1791 | val_acc: 61.5000% , val_loss: 1.3706 | learning_rate: 0.00050

Training epoch 30...
Epoch 30: train_acc: 65.5600% , train_loss: 1.1662 | val_acc: 62.8200% , val_loss: 1.3333 | learning_rate: 0.00050

Training finished
Restored Best weights from epoch: 30



11th run:

Changed epochs to 60



11th run results:

Choose epochs: 60
Choose weights' initialization for Convolutional Layers (He or Xavier): he
Choose weights' distribution for Convolutional Layers (Uniform or Normal): normal
Choose learning rate: 0.001
Choose optimizer:
1) SGD
2) SGD_momentum
3) Adam
4) AdamW
3
Use data augmentation? (yes/no): y

Training epoch 1...
Epoch 1: train_acc: 14.1067% , train_loss: 3.6662 | val_acc: 19.2800% , val_loss: 3.4311 | learning_rate: 0.00100

Training epoch 2...
Epoch 2: train_acc: 26.3156% , train_loss: 2.9317 | val_acc: 31.3000% , val_loss: 2.7324 | learning_rate: 0.00100

Training epoch 3...
Epoch 3: train_acc: 33.2844% , train_loss: 2.5705 | val_acc: 34.9400% , val_loss: 2.5383 | learning_rate: 0.00100

Training epoch 4...
Epoch 4: train_acc: 38.3756% , train_loss: 2.3464 | val_acc: 42.4000% , val_loss: 2.1557 | learning_rate: 0.00100

Training epoch 5...
Epoch 5: train_acc: 41.2844% , train_loss: 2.1980 | val_acc: 43.6800% , val_loss: 2.1148 | learning_rate: 0.00100

Training epoch 6...
Epoch 6: train_acc: 44.3622% , train_loss: 2.0642 | val_acc: 46.3600% , val_loss: 1.9906 | learning_rate: 0.00100

Training epoch 7...
Epoch 7: train_acc: 46.2733% , train_loss: 1.9755 | val_acc: 45.5000% , val_loss: 2.0362 | learning_rate: 0.00100

Training epoch 8...
Epoch 8: train_acc: 48.0578% , train_loss: 1.8964 | val_acc: 48.6800% , val_loss: 1.9507 | learning_rate: 0.00100

Training epoch 9...
Epoch 9: train_acc: 49.6422% , train_loss: 1.8260 | val_acc: 49.5200% , val_loss: 1.8738 | learning_rate: 0.00100

Training epoch 10...
Epoch 10: train_acc: 51.0089% , train_loss: 1.7733 | val_acc: 53.2600% , val_loss: 1.7191 | learning_rate: 0.00100

Training epoch 11...
Epoch 11: train_acc: 52.1311% , train_loss: 1.7112 | val_acc: 53.0800% , val_loss: 1.6885 | learning_rate: 0.00100

Training epoch 12...
Epoch 12: train_acc: 53.0711% , train_loss: 1.6739 | val_acc: 54.0000% , val_loss: 1.6646 | learning_rate: 0.00100

Training epoch 13...
Epoch 13: train_acc: 54.3022% , train_loss: 1.6308 | val_acc: 54.8400% , val_loss: 1.6303 | learning_rate: 0.00100

Training epoch 14...
Epoch 14: train_acc: 55.2156% , train_loss: 1.6022 | val_acc: 54.0200% , val_loss: 1.6829 | learning_rate: 0.00100

Training epoch 15...
Epoch 15: train_acc: 55.9400% , train_loss: 1.5654 | val_acc: 55.7800% , val_loss: 1.5995 | learning_rate: 0.00100

Training epoch 16...
Epoch 16: train_acc: 56.4067% , train_loss: 1.5430 | val_acc: 56.7200% , val_loss: 1.5473 | learning_rate: 0.00100

Training epoch 17...
Epoch 17: train_acc: 56.6511% , train_loss: 1.5191 | val_acc: 54.4800% , val_loss: 1.6595 | learning_rate: 0.00100

Training epoch 18...
Epoch 18: train_acc: 57.5978% , train_loss: 1.4833 | val_acc: 56.2600% , val_loss: 1.5753 | learning_rate: 0.00100

Training epoch 19...
Epoch 19: train_acc: 58.5356% , train_loss: 1.4619 | val_acc: 57.4800% , val_loss: 1.5012 | learning_rate: 0.00100

Training epoch 20...
Epoch 20: train_acc: 58.9178% , train_loss: 1.4375 | val_acc: 58.4200% , val_loss: 1.5081 | learning_rate: 0.00100

Training epoch 21...
Epoch 21: train_acc: 59.3111% , train_loss: 1.4227 | val_acc: 57.8600% , val_loss: 1.5345 | learning_rate: 0.00100

Training epoch 22...
Epoch 22: train_acc: 60.0778% , train_loss: 1.3924 | val_acc: 59.3600% , val_loss: 1.4794 | learning_rate: 0.00100

Training epoch 23...
Epoch 23: train_acc: 60.6356% , train_loss: 1.3748 | val_acc: 57.7600% , val_loss: 1.5346 | learning_rate: 0.00100

Training epoch 24...
Epoch 24: train_acc: 60.8556% , train_loss: 1.3584 | val_acc: 58.0000% , val_loss: 1.5149 | learning_rate: 0.00100

Training epoch 25...
Epoch 25: train_acc: 61.6622% , train_loss: 1.3380 | val_acc: 57.3000% , val_loss: 1.5410 | learning_rate: 0.00100

Training epoch 26...
Epoch 26: train_acc: 61.8022% , train_loss: 1.3204 | val_acc: 58.3200% , val_loss: 1.5089 | learning_rate: 0.00100

Training epoch 27...
Epoch 27: train_acc: 64.2289% , train_loss: 1.2300 | val_acc: 61.9800% , val_loss: 1.3641 | learning_rate: 0.00050

Training epoch 28...
Epoch 28: train_acc: 64.7689% , train_loss: 1.2016 | val_acc: 61.3000% , val_loss: 1.4004 | learning_rate: 0.00050

Training epoch 29...
Epoch 29: train_acc: 65.3956% , train_loss: 1.1791 | val_acc: 61.5000% , val_loss: 1.3706 | learning_rate: 0.00050

Training epoch 30...
Epoch 30: train_acc: 65.5600% , train_loss: 1.1662 | val_acc: 62.8200% , val_loss: 1.3333 | learning_rate: 0.00050

Training epoch 31...
Epoch 31: train_acc: 66.2867% , train_loss: 1.1547 | val_acc: 62.2800% , val_loss: 1.3807 | learning_rate: 0.00050

Training epoch 32...
Epoch 32: train_acc: 66.4222% , train_loss: 1.1329 | val_acc: 62.8600% , val_loss: 1.3519 | learning_rate: 0.00050

Training epoch 33...
Epoch 33: train_acc: 66.7467% , train_loss: 1.1344 | val_acc: 63.1600% , val_loss: 1.3358 | learning_rate: 0.00050

Training epoch 34...
Epoch 34: train_acc: 66.8733% , train_loss: 1.1219 | val_acc: 62.0000% , val_loss: 1.3944 | learning_rate: 0.00050

Training epoch 35...
Epoch 35: train_acc: 68.0956% , train_loss: 1.0719 | val_acc: 63.6600% , val_loss: 1.2923 | learning_rate: 0.00025

Training epoch 36...
Epoch 36: train_acc: 68.3911% , train_loss: 1.0548 | val_acc: 63.6000% , val_loss: 1.3001 | learning_rate: 0.00025

Training epoch 37...
Epoch 37: train_acc: 68.8933% , train_loss: 1.0519 | val_acc: 63.6600% , val_loss: 1.3039 | learning_rate: 0.00025

Training epoch 38...
Epoch 38: train_acc: 68.7067% , train_loss: 1.0461 | val_acc: 64.0400% , val_loss: 1.3020 | learning_rate: 0.00025

Training epoch 39...
Epoch 39: train_acc: 69.1089% , train_loss: 1.0324 | val_acc: 63.5600% , val_loss: 1.3096 | learning_rate: 0.00025

Training epoch 40...
Epoch 40: train_acc: 69.9622% , train_loss: 1.0076 | val_acc: 63.7000% , val_loss: 1.2825 | learning_rate: 0.00013

Training epoch 41...
Epoch 41: train_acc: 69.6867% , train_loss: 1.0073 | val_acc: 64.3400% , val_loss: 1.2786 | learning_rate: 0.00013

Training epoch 42...
Epoch 42: train_acc: 70.0644% , train_loss: 0.9994 | val_acc: 64.1800% , val_loss: 1.2865 | learning_rate: 0.00013

Training epoch 43...
Epoch 43: train_acc: 70.3444% , train_loss: 0.9911 | val_acc: 64.0600% , val_loss: 1.2833 | learning_rate: 0.00013

Training epoch 44...
Epoch 44: train_acc: 70.3356% , train_loss: 0.9913 | val_acc: 64.5800% , val_loss: 1.2803 | learning_rate: 0.00013

Training epoch 45...
Epoch 45: train_acc: 70.3200% , train_loss: 0.9865 | val_acc: 64.0000% , val_loss: 1.2798 | learning_rate: 0.00013

Training epoch 46...
Epoch 46: train_acc: 70.7111% , train_loss: 0.9768 | val_acc: 64.2200% , val_loss: 1.2770 | learning_rate: 0.00006

Training epoch 47...
Epoch 47: train_acc: 71.0533% , train_loss: 0.9703 | val_acc: 64.4000% , val_loss: 1.2761 | learning_rate: 0.00006

Training epoch 48...
Epoch 48: train_acc: 71.1089% , train_loss: 0.9588 | val_acc: 64.6200% , val_loss: 1.2730 | learning_rate: 0.00006

Training epoch 49...
Epoch 49: train_acc: 71.2733% , train_loss: 0.9645 | val_acc: 64.6000% , val_loss: 1.2707 | learning_rate: 0.00006

Training epoch 50...
Epoch 50: train_acc: 71.0756% , train_loss: 0.9612 | val_acc: 64.6200% , val_loss: 1.2703 | learning_rate: 0.00006

Training epoch 51...
Epoch 51: train_acc: 71.5444% , train_loss: 0.9528 | val_acc: 64.9800% , val_loss: 1.2747 | learning_rate: 0.00006

Training epoch 52...
Epoch 52: train_acc: 71.2689% , train_loss: 0.9564 | val_acc: 64.7000% , val_loss: 1.2693 | learning_rate: 0.00006

Training epoch 53...
Epoch 53: train_acc: 71.2244% , train_loss: 0.9587 | val_acc: 64.6400% , val_loss: 1.2746 | learning_rate: 0.00006

Training epoch 54...
Epoch 54: train_acc: 71.4556% , train_loss: 0.9534 | val_acc: 64.6800% , val_loss: 1.2705 | learning_rate: 0.00006

Training epoch 55...
Epoch 55: train_acc: 71.1867% , train_loss: 0.9561 | val_acc: 64.5400% , val_loss: 1.2694 | learning_rate: 0.00006

Training epoch 56...
Epoch 56: train_acc: 71.4133% , train_loss: 0.9467 | val_acc: 64.5600% , val_loss: 1.2739 | learning_rate: 0.00006

Training epoch 57...
Epoch 57: train_acc: 71.4933% , train_loss: 0.9426 | val_acc: 64.7000% , val_loss: 1.2715 | learning_rate: 0.00003

Training epoch 58...
Epoch 58: train_acc: 71.7000% , train_loss: 0.9370 | val_acc: 64.6000% , val_loss: 1.2700 | learning_rate: 0.00003

Training epoch 59...
Epoch 59: train_acc: 72.0533% , train_loss: 0.9302 | val_acc: 64.5800% , val_loss: 1.2712 | learning_rate: 0.00003

Training epoch 60...
Epoch 60: train_acc: 71.8933% , train_loss: 0.9350 | val_acc: 64.5000% , val_loss: 1.2697 | learning_rate: 0.00003

Training finished
Restored Best weights from epoch: 52



12th run:

changed dropout rate from 0.5 to 0.25



12th run results:

Training epoch 1...
Epoch 1: train_acc: 16.2756% , train_loss: 3.5406 | val_acc: 23.4600% , val_loss: 3.1499 | learning_rate: 0.00100

Training epoch 2...
Epoch 2: train_acc: 29.8667% , train_loss: 2.7609 | val_acc: 35.0400% , val_loss: 2.5578 | learning_rate: 0.00100

Training epoch 3...
Epoch 3: train_acc: 37.4889% , train_loss: 2.3913 | val_acc: 39.3400% , val_loss: 2.3100 | learning_rate: 0.00100

Training epoch 4...
Epoch 4: train_acc: 42.5844% , train_loss: 2.1625 | val_acc: 44.4800% , val_loss: 2.0547 | learning_rate: 0.00100

Training epoch 5...
Epoch 5: train_acc: 46.0644% , train_loss: 2.0070 | val_acc: 46.2800% , val_loss: 2.0034 | learning_rate: 0.00100

Training epoch 6...
Epoch 6: train_acc: 48.7467% , train_loss: 1.8781 | val_acc: 48.6800% , val_loss: 1.9100 | learning_rate: 0.00100

Training epoch 7...
Epoch 7: train_acc: 51.1489% , train_loss: 1.7820 | val_acc: 44.5800% , val_loss: 2.1061 | learning_rate: 0.00100

Training epoch 8...
Epoch 8: train_acc: 52.7378% , train_loss: 1.7093 | val_acc: 52.7800% , val_loss: 1.7635 | learning_rate: 0.00100

Training epoch 9...
Epoch 9: train_acc: 54.4444% , train_loss: 1.6365 | val_acc: 53.1000% , val_loss: 1.7323 | learning_rate: 0.00100

Training epoch 10...
Epoch 10: train_acc: 55.8356% , train_loss: 1.5778 | val_acc: 53.1000% , val_loss: 1.7147 | learning_rate: 0.00100

Training epoch 11...
Epoch 11: train_acc: 57.0044% , train_loss: 1.5253 | val_acc: 53.9000% , val_loss: 1.6711 | learning_rate: 0.00100

Training epoch 12...
Epoch 12: train_acc: 58.1956% , train_loss: 1.4759 | val_acc: 55.0200% , val_loss: 1.6217 | learning_rate: 0.00100

Training epoch 13...
Epoch 13: train_acc: 59.4822% , train_loss: 1.4282 | val_acc: 54.4400% , val_loss: 1.6817 | learning_rate: 0.00100

Training epoch 14...
Epoch 14: train_acc: 60.3556% , train_loss: 1.3907 | val_acc: 55.5400% , val_loss: 1.6403 | learning_rate: 0.00100

Training epoch 15...
Epoch 15: train_acc: 61.5267% , train_loss: 1.3502 | val_acc: 54.2800% , val_loss: 1.6889 | learning_rate: 0.00100

Training epoch 16...
Epoch 16: train_acc: 62.0844% , train_loss: 1.3240 | val_acc: 56.9000% , val_loss: 1.5996 | learning_rate: 0.00100

Training epoch 17...
Epoch 17: train_acc: 62.9222% , train_loss: 1.2944 | val_acc: 57.4600% , val_loss: 1.5480 | learning_rate: 0.00100

Training epoch 18...
Epoch 18: train_acc: 63.4400% , train_loss: 1.2587 | val_acc: 58.5400% , val_loss: 1.5226 | learning_rate: 0.00100

Training epoch 19...
Epoch 19: train_acc: 64.2311% , train_loss: 1.2294 | val_acc: 56.6000% , val_loss: 1.5836 | learning_rate: 0.00100

Training epoch 20...
Epoch 20: train_acc: 64.9000% , train_loss: 1.2098 | val_acc: 58.1200% , val_loss: 1.5489 | learning_rate: 0.00100

Training epoch 21...
Epoch 21: train_acc: 65.0800% , train_loss: 1.1893 | val_acc: 59.8800% , val_loss: 1.4694 | learning_rate: 0.00100

Training epoch 22...
Epoch 22: train_acc: 65.8844% , train_loss: 1.1640 | val_acc: 59.1400% , val_loss: 1.5104 | learning_rate: 0.00100

Training epoch 23...
Epoch 23: train_acc: 66.5578% , train_loss: 1.1386 | val_acc: 57.9000% , val_loss: 1.5585 | learning_rate: 0.00100

Training epoch 24...
Epoch 24: train_acc: 67.2356% , train_loss: 1.1227 | val_acc: 58.0200% , val_loss: 1.6187 | learning_rate: 0.00100

Training epoch 25...
Epoch 25: train_acc: 67.1889% , train_loss: 1.1068 | val_acc: 59.4800% , val_loss: 1.5084 | learning_rate: 0.00100

Training epoch 26...
Epoch 26: train_acc: 70.5289% , train_loss: 0.9852 | val_acc: 62.0000% , val_loss: 1.4028 | learning_rate: 0.00050

Training epoch 27...
Epoch 27: train_acc: 71.5444% , train_loss: 0.9521 | val_acc: 62.4800% , val_loss: 1.3982 | learning_rate: 0.00050

Training epoch 28...
Epoch 28: train_acc: 71.9089% , train_loss: 0.9380 | val_acc: 62.6400% , val_loss: 1.3809 | learning_rate: 0.00050

Training epoch 29...
Epoch 29: train_acc: 72.6444% , train_loss: 0.9164 | val_acc: 62.7800% , val_loss: 1.3723 | learning_rate: 0.00050

Training epoch 30...
Epoch 30: train_acc: 72.8444% , train_loss: 0.8981 | val_acc: 62.0600% , val_loss: 1.4205 | learning_rate: 0.00050

Training epoch 31...
Epoch 31: train_acc: 73.0333% , train_loss: 0.8942 | val_acc: 61.8600% , val_loss: 1.3985 | learning_rate: 0.00050

Training epoch 32...
Epoch 32: train_acc: 73.3911% , train_loss: 0.8786 | val_acc: 62.4400% , val_loss: 1.4081 | learning_rate: 0.00050

Training epoch 33...
Epoch 33: train_acc: 73.7956% , train_loss: 0.8691 | val_acc: 62.9200% , val_loss: 1.3938 | learning_rate: 0.00050

Training epoch 34...
Epoch 34: train_acc: 75.3378% , train_loss: 0.8131 | val_acc: 63.5400% , val_loss: 1.3508 | learning_rate: 0.00025

Training epoch 35...
Epoch 35: train_acc: 75.6222% , train_loss: 0.7971 | val_acc: 64.3000% , val_loss: 1.3470 | learning_rate: 0.00025

Training epoch 36...
Epoch 36: train_acc: 76.4822% , train_loss: 0.7684 | val_acc: 63.9000% , val_loss: 1.3550 | learning_rate: 0.00025

Training epoch 37...
Epoch 37: train_acc: 76.4578% , train_loss: 0.7753 | val_acc: 63.8600% , val_loss: 1.3762 | learning_rate: 0.00025

Training epoch 38...
Epoch 38: train_acc: 76.4489% , train_loss: 0.7695 | val_acc: 64.0000% , val_loss: 1.3641 | learning_rate: 0.00025

Training epoch 39...
Epoch 39: train_acc: 76.7733% , train_loss: 0.7592 | val_acc: 63.7600% , val_loss: 1.3583 | learning_rate: 0.00025

Training epoch 40...
Epoch 40: train_acc: 77.7467% , train_loss: 0.7276 | val_acc: 64.2800% , val_loss: 1.3437 | learning_rate: 0.00013

Training epoch 41...
Epoch 41: train_acc: 78.0911% , train_loss: 0.7193 | val_acc: 63.9400% , val_loss: 1.3394 | learning_rate: 0.00013

Training epoch 42...
Epoch 42: train_acc: 77.9533% , train_loss: 0.7161 | val_acc: 64.3400% , val_loss: 1.3505 | learning_rate: 0.00013

Training epoch 43...
Epoch 43: train_acc: 78.1578% , train_loss: 0.7123 | val_acc: 64.3200% , val_loss: 1.3454 | learning_rate: 0.00013

Training epoch 44...




13th run:

Changed Dropout position: From after Flatten layer -> To after ReLU of Dense(256)



13th run results:


Training epoch 1...
Epoch 1: train_acc: 15.4978% , train_loss: 3.5859 | val_acc: 20.7200% , val_loss: 3.2370 | learning_rate: 0.00100

Training epoch 2...
Epoch 2: train_acc: 28.7733% , train_loss: 2.8212 | val_acc: 33.9400% , val_loss: 2.5297 | learning_rate: 0.00100

Training epoch 3...
Epoch 3: train_acc: 35.6378% , train_loss: 2.4741 | val_acc: 41.3200% , val_loss: 2.2395 | learning_rate: 0.00100

Training epoch 4...
Epoch 4: train_acc: 40.3000% , train_loss: 2.2542 | val_acc: 43.8000% , val_loss: 2.1309 | learning_rate: 0.00100

Training epoch 5...
Epoch 5: train_acc: 44.2200% , train_loss: 2.0858 | val_acc: 47.3400% , val_loss: 1.9734 | learning_rate: 0.00100

Training epoch 6...
Epoch 6: train_acc: 46.7578% , train_loss: 1.9634 | val_acc: 44.3200% , val_loss: 2.0855 | learning_rate: 0.00100

Training epoch 7...
Epoch 7: train_acc: 48.7622% , train_loss: 1.8736 | val_acc: 48.0600% , val_loss: 1.9421 | learning_rate: 0.00100

Training epoch 8...
Epoch 8: train_acc: 50.5711% , train_loss: 1.7957 | val_acc: 52.3800% , val_loss: 1.7578 | learning_rate: 0.00100

Training epoch 9...
Epoch 9: train_acc: 52.3778% , train_loss: 1.7239 | val_acc: 53.4800% , val_loss: 1.7335 | learning_rate: 0.00100

Training epoch 10...
Epoch 10: train_acc: 53.9511% , train_loss: 1.6587 | val_acc: 52.2200% , val_loss: 1.7651 | learning_rate: 0.00100

Training epoch 11...
Epoch 11: train_acc: 54.9444% , train_loss: 1.6014 | val_acc: 53.5400% , val_loss: 1.6897 | learning_rate: 0.00100

Training epoch 12...
Epoch 12: train_acc: 56.2356% , train_loss: 1.5612 | val_acc: 53.2200% , val_loss: 1.7346 | learning_rate: 0.00100

Training epoch 13...
Epoch 13: train_acc: 57.2422% , train_loss: 1.5088 | val_acc: 53.1000% , val_loss: 1.7338 | learning_rate: 0.00100

Training epoch 14...
Epoch 14: train_acc: 58.2644% , train_loss: 1.4750 | val_acc: 55.6800% , val_loss: 1.5981 | learning_rate: 0.00100

Training epoch 15...
Epoch 15: train_acc: 59.3578% , train_loss: 1.4239 | val_acc: 56.3600% , val_loss: 1.5774 | learning_rate: 0.00100

Training epoch 16...
Epoch 16: train_acc: 60.0222% , train_loss: 1.3974 | val_acc: 57.3400% , val_loss: 1.5670 | learning_rate: 0.00100

Training epoch 17...
Epoch 17: train_acc: 60.6000% , train_loss: 1.3695 | val_acc: 56.6600% , val_loss: 1.5767 | learning_rate: 0.00100

Training epoch 18...
Epoch 18: train_acc: 61.2600% , train_loss: 1.3420 | val_acc: 57.8800% , val_loss: 1.5760 | learning_rate: 0.00100

Training epoch 19...
Epoch 19: train_acc: 62.5467% , train_loss: 1.3015 | val_acc: 58.4800% , val_loss: 1.5325 | learning_rate: 0.00100

Training epoch 20...
Epoch 20: train_acc: 62.7600% , train_loss: 1.2828 | val_acc: 57.1200% , val_loss: 1.6588 | learning_rate: 0.00100

Training epoch 21...
Epoch 21: train_acc: 63.7244% , train_loss: 1.2536 | val_acc: 57.8400% , val_loss: 1.5519 | learning_rate: 0.00100

Training epoch 22...
Epoch 22: train_acc: 63.9267% , train_loss: 1.2347 | val_acc: 57.6600% , val_loss: 1.5743 | learning_rate: 0.00100

Training epoch 23...
Epoch 23: train_acc: 64.7644% , train_loss: 1.2113 | val_acc: 59.1400% , val_loss: 1.5266 | learning_rate: 0.00100

Training epoch 24...
Epoch 24: train_acc: 65.0089% , train_loss: 1.1970 | val_acc: 57.4000% , val_loss: 1.5691 | learning_rate: 0.00100

Training epoch 25...
Epoch 25: train_acc: 65.5000% , train_loss: 1.1779 | val_acc: 60.1600% , val_loss: 1.4794 | learning_rate: 0.00100

Training epoch 26...
Epoch 26: train_acc: 66.0356% , train_loss: 1.1558 | val_acc: 58.6400% , val_loss: 1.5406 | learning_rate: 0.00100

Training epoch 27...
Epoch 27: train_acc: 66.5578% , train_loss: 1.1378 | val_acc: 58.6400% , val_loss: 1.5688 | learning_rate: 0.00100

Training epoch 28...
Epoch 28: train_acc: 67.0978% , train_loss: 1.1219 | val_acc: 59.8000% , val_loss: 1.5068 | learning_rate: 0.00100

Training epoch 29...
Epoch 29: train_acc: 67.5244% , train_loss: 1.1033 | val_acc: 60.3200% , val_loss: 1.4815 | learning_rate: 0.00100

Training epoch 30...
Epoch 30: train_acc: 70.1044% , train_loss: 1.0084 | val_acc: 61.8800% , val_loss: 1.4249 | learning_rate: 0.00050

Training epoch 31...
Epoch 31: train_acc: 71.1267% , train_loss: 0.9615 | val_acc: 63.9000% , val_loss: 1.3798 | learning_rate: 0.00050

Training epoch 32...
Epoch 32: train_acc: 71.5578% , train_loss: 0.9427 | val_acc: 62.6200% , val_loss: 1.4113 | learning_rate: 0.00050

Training epoch 33...
Epoch 33: train_acc: 72.0800% , train_loss: 0.9295 | val_acc: 63.1800% , val_loss: 1.3929 | learning_rate: 0.00050

Training epoch 34...
Epoch 34: train_acc: 72.4467% , train_loss: 0.9087 | val_acc: 62.9800% , val_loss: 1.4110 | learning_rate: 0.00050

Training epoch 35...
Epoch 35: train_acc: 72.8578% , train_loss: 0.9004 | val_acc: 63.5600% , val_loss: 1.3873 | learning_rate: 0.00050

Training epoch 36...
Epoch 36: train_acc: 74.3156% , train_loss: 0.8442 | val_acc: 64.8400% , val_loss: 1.3431 | learning_rate: 0.00025

Training epoch 37...
Epoch 37: train_acc: 74.7778% , train_loss: 0.8283 | val_acc: 64.0800% , val_loss: 1.3770 | learning_rate: 0.00025

Training epoch 38...
Epoch 38: train_acc: 75.1711% , train_loss: 0.8190 | val_acc: 64.5800% , val_loss: 1.3648 | learning_rate: 0.00025

Training epoch 39...
Epoch 39: train_acc: 75.1467% , train_loss: 0.8148 | val_acc: 64.1800% , val_loss: 1.3540 | learning_rate: 0.00025

Training epoch 40...
Epoch 40: train_acc: 75.4378% , train_loss: 0.8003 | val_acc: 64.0800% , val_loss: 1.3683 | learning_rate: 0.00025

Training epoch 41...
Epoch 41: train_acc: 76.0489% , train_loss: 0.7763 | val_acc: 64.8200% , val_loss: 1.3425 | learning_rate: 0.00013

Training epoch 42...
Epoch 42: train_acc: 76.2889% , train_loss: 0.7665 | val_acc: 64.7000% , val_loss: 1.3448 | learning_rate: 0.00013

Training epoch 43...
Epoch 43: train_acc: 76.7089% , train_loss: 0.7623 | val_acc: 64.8200% , val_loss: 1.3697 | learning_rate: 0.00013

Training epoch 44...
Epoch 44: train_acc: 76.7444% , train_loss: 0.7493 | val_acc: 64.8600% , val_loss: 1.3568 | learning_rate: 0.00013

Training finished
Restored Best weights from epoch: 36



14th run:

Added Dense(128) + BatchNorm + ReLU after Dense(256)



14th run results:


Training epoch 1...
Epoch 1: train_acc: 13.7889% , train_loss: 3.6988 | val_acc: 23.0400% , val_loss: 3.1070 | learning_rate: 0.00100

Training epoch 2...
Epoch 2: train_acc: 24.9133% , train_loss: 2.9834 | val_acc: 28.3000% , val_loss: 2.8466 | learning_rate: 0.00100

Training epoch 3...
Epoch 3: train_acc: 32.3978% , train_loss: 2.6046 | val_acc: 37.7800% , val_loss: 2.3324 | learning_rate: 0.00100

Training epoch 4...
Epoch 4: train_acc: 37.2622% , train_loss: 2.3728 | val_acc: 42.5400% , val_loss: 2.1564 | learning_rate: 0.00100

Training epoch 5...
Epoch 5: train_acc: 41.1622% , train_loss: 2.2060 | val_acc: 43.0200% , val_loss: 2.1515 | learning_rate: 0.00100

Training epoch 6...
Epoch 6: train_acc: 43.7178% , train_loss: 2.0768 | val_acc: 45.7000% , val_loss: 2.0412 | learning_rate: 0.00100

Training epoch 7...
Epoch 7: train_acc: 46.1333% , train_loss: 1.9840 | val_acc: 47.0400% , val_loss: 1.9956 | learning_rate: 0.00100

Training epoch 8...
Epoch 8: train_acc: 48.0311% , train_loss: 1.8947 | val_acc: 47.7400% , val_loss: 1.9901 | learning_rate: 0.00100

Training epoch 9...
Epoch 9: train_acc: 49.4289% , train_loss: 1.8222 | val_acc: 51.3800% , val_loss: 1.7518 | learning_rate: 0.00100

Training epoch 10...
Epoch 10: train_acc: 51.1089% , train_loss: 1.7654 | val_acc: 50.7800% , val_loss: 1.8244 | learning_rate: 0.00100

Training epoch 11...
Epoch 11: train_acc: 52.5444% , train_loss: 1.7016 | val_acc: 51.8400% , val_loss: 1.7612 | learning_rate: 0.00100

Training epoch 12...
Epoch 12: train_acc: 53.4800% , train_loss: 1.6611 | val_acc: 50.2200% , val_loss: 1.8565 | learning_rate: 0.00100

Training epoch 13...
Epoch 13: train_acc: 54.5956% , train_loss: 1.6099 | val_acc: 51.5800% , val_loss: 1.7586 | learning_rate: 0.00100

Training epoch 14...
Epoch 14: train_acc: 57.9533% , train_loss: 1.4797 | val_acc: 58.6600% , val_loss: 1.4767 | learning_rate: 0.00050

Training epoch 15...
Epoch 15: train_acc: 58.8956% , train_loss: 1.4316 | val_acc: 56.8800% , val_loss: 1.5355 | learning_rate: 0.00050

Training epoch 16...
Epoch 16: train_acc: 59.9622% , train_loss: 1.3963 | val_acc: 57.6200% , val_loss: 1.4921 | learning_rate: 0.00050

Training epoch 17...
Epoch 17: train_acc: 60.8356% , train_loss: 1.3705 | val_acc: 58.3600% , val_loss: 1.4849 | learning_rate: 0.00050

Training epoch 18...
Epoch 18: train_acc: 61.3933% , train_loss: 1.3397 | val_acc: 59.3000% , val_loss: 1.4567 | learning_rate: 0.00050

Training epoch 19...
Epoch 19: train_acc: 61.8222% , train_loss: 1.3162 | val_acc: 58.3800% , val_loss: 1.4987 | learning_rate: 0.00050

Training epoch 20...
Epoch 20: train_acc: 62.3333% , train_loss: 1.2954 | val_acc: 59.6800% , val_loss: 1.4464 | learning_rate: 0.00050

Training epoch 21...
Epoch 21: train_acc: 62.4956% , train_loss: 1.2865 | val_acc: 59.0400% , val_loss: 1.4863 | learning_rate: 0.00050

Training epoch 22...
Epoch 22: train_acc: 63.3111% , train_loss: 1.2595 | val_acc: 59.4600% , val_loss: 1.4895 | learning_rate: 0.00050

Training epoch 23...
Epoch 23: train_acc: 63.2978% , train_loss: 1.2495 | val_acc: 59.9000% , val_loss: 1.4484 | learning_rate: 0.00050

Training epoch 24...
Epoch 24: train_acc: 64.4067% , train_loss: 1.2190 | val_acc: 59.1600% , val_loss: 1.4709 | learning_rate: 0.00050

Training epoch 25...
Epoch 25: train_acc: 66.2444% , train_loss: 1.1553 | val_acc: 61.2400% , val_loss: 1.3963 | learning_rate: 0.00025

Training epoch 26...
Epoch 26: train_acc: 66.7400% , train_loss: 1.1374 | val_acc: 60.9400% , val_loss: 1.3934 | learning_rate: 0.00025

Training epoch 27...
Epoch 27: train_acc: 66.9667% , train_loss: 1.1160 | val_acc: 61.4800% , val_loss: 1.3896 | learning_rate: 0.00025

Training epoch 28...
Epoch 28: train_acc: 67.1667% , train_loss: 1.1028 | val_acc: 61.3800% , val_loss: 1.3987 | learning_rate: 0.00025

Training epoch 29...
Epoch 29: train_acc: 67.4867% , train_loss: 1.1010 | val_acc: 61.5200% , val_loss: 1.4041 | learning_rate: 0.00025

Training epoch 30...
Epoch 30: train_acc: 67.9000% , train_loss: 1.0822 | val_acc: 61.6400% , val_loss: 1.4069 | learning_rate: 0.00025

Training epoch 31...
Epoch 31: train_acc: 67.8600% , train_loss: 1.0849 | val_acc: 61.2400% , val_loss: 1.4002 | learning_rate: 0.00025

Training epoch 32...
Epoch 32: train_acc: 69.2889% , train_loss: 1.0384 | val_acc: 61.8800% , val_loss: 1.3706 | learning_rate: 0.00013

Training epoch 33...
Epoch 33: train_acc: 69.3000% , train_loss: 1.0277 | val_acc: 62.2600% , val_loss: 1.3616 | learning_rate: 0.00013

Training epoch 34...
Epoch 34: train_acc: 69.6022% , train_loss: 1.0162 | val_acc: 62.3000% , val_loss: 1.3680 | learning_rate: 0.00013

Training epoch 35...
Epoch 35: train_acc: 69.8489% , train_loss: 1.0138 | val_acc: 62.8000% , val_loss: 1.3622 | learning_rate: 0.00013

Training epoch 36...
Epoch 36: train_acc: 69.6778% , train_loss: 1.0116 | val_acc: 62.0400% , val_loss: 1.3679 | learning_rate: 0.00013

Training epoch 37...
Epoch 37: train_acc: 70.1156% , train_loss: 0.9950 | val_acc: 62.1200% , val_loss: 1.3648 | learning_rate: 0.00013

Training epoch 38...
Epoch 38: train_acc: 70.7867% , train_loss: 0.9739 | val_acc: 61.8600% , val_loss: 1.3625 | learning_rate: 0.00006

Training epoch 39...
Epoch 39: train_acc: 70.7711% , train_loss: 0.9737 | val_acc: 62.6200% , val_loss: 1.3539 | learning_rate: 0.00006

Training epoch 40...
Epoch 40: train_acc: 70.9378% , train_loss: 0.9753 | val_acc: 62.3000% , val_loss: 1.3635 | learning_rate: 0.00006

Training epoch 41...
Epoch 41: train_acc: 70.9867% , train_loss: 0.9651 | val_acc: 62.3800% , val_loss: 1.3636 | learning_rate: 0.00006

Training epoch 42...
Epoch 42: train_acc: 70.8822% , train_loss: 0.9633 | val_acc: 62.7600% , val_loss: 1.3607 | learning_rate: 0.00006

Training epoch 43...
Epoch 43: train_acc: 71.0089% , train_loss: 0.9579 | val_acc: 62.2600% , val_loss: 1.3614 | learning_rate: 0.00006

Training epoch 44...
Epoch 44: train_acc: 71.1689% , train_loss: 0.9528 | val_acc: 62.4000% , val_loss: 1.3572 | learning_rate: 0.00003

Training epoch 45...
Epoch 45: train_acc: 71.4644% , train_loss: 0.9512 | val_acc: 62.3600% , val_loss: 1.3571 | learning_rate: 0.00003

Training epoch 46...
Epoch 46: train_acc: 71.0556% , train_loss: 0.9535 | val_acc: 62.8400% , val_loss: 1.3564 | learning_rate: 0.00003

Training epoch 47...
Epoch 47: train_acc: 71.3800% , train_loss: 0.9541 | val_acc: 62.5200% , val_loss: 1.3550 | learning_rate: 0.00003

Training finished
Restored Best weights from epoch: 39