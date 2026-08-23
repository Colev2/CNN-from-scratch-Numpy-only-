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

