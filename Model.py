from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

CSV_COLUMN_NAMES = ['Behavior', 'GPA', 'Gender', 'Age', 'Parents', 'GoodAttendance']
Behavior = ['Good', 'Bad', 'Normal']

train = pd.read_csv('converted_Data.csv', names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv('converted_NewData.csv', names=CSV_COLUMN_NAMES, header=0)

print(train.head())
train_y = train.pop('Behavior')
test_y = test.pop('Behavior')

def input_fn(features, labels, training=True, batch_size=256):

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,

    hidden_units=[40, 15],

    n_classes=3)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=10000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}%\n'.format(accuracy=eval_result['accuracy'] * 100))


predict_input_fn = lambda: input_fn(test, test_y, training=False)
predictions = list(classifier.predict(predict_input_fn))

predicted_labels = [prediction['class_ids'][0] for prediction in predictions]


behavior_counts = train_y.value_counts()

predicted_behavior_counts = pd.Series(predicted_labels).value_counts()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

unique_labels = train_y.unique()

manual_labels = [str(label) for label in unique_labels]

axs[0].pie(behavior_counts, labels=manual_labels, autopct='%1.1f%%', startangle=90)
axs[0].axis('equal')
axs[0].set_title('Distribution of Original Student Behaviors')

axs[1].pie(predicted_behavior_counts, labels=manual_labels, autopct='%1.1f%%', startangle=90)
axs[1].axis('equal')
axs[1].set_xticks(range(len(unique_labels)))
axs[1].set_title('Distribution of Predicted Student Behaviors')

plt.tight_layout()
plt.show()
