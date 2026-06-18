
CLASSES = np.array([ 'airplane', 'automobile', 
                    'bird', 'cat', 'deer', 'dog',
                      'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_trest)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]