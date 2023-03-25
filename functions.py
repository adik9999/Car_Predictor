def plot_loss(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  epoch = np.arange(1,6)
  plt.figure(figsize=(15,6))
  plt.subplot(1,2,1)
  plt.title("LOSS CURVE")
  plt.plot(epoch,val_loss,label='validation_loss')
  plt.plot(epoch,loss,label='training_loss')
  plt.legend()
  # plt.subplot(1,3,2)
  # plt.title("LOSS")
  # plt.plot(epoch,loss)
  plt.subplot(1,2,2)
  plt.title("ACCURACY")
  plt.plot(epoch,accuracy,label='training_accuracy')
  plt.plot(epoch,val_accuracy,label='validation_accuracy')
  plt.legend()

def tensorboard_callback(dir_name,experiment):
  log_dir = dir_name + "/" + experiment + "/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving Tensorboard log to : {log_dir}")
  return tensorboard_callback
