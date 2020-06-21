# -*-coding:utf-8-*-
import os
import struct
import collections
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

train_file = "D:\python project me\data\my_point_net/train/train.txt"
val_file = "D:\python project me\data\my_point_net/val/val.txt"
test_file = "D:\python project me\data\my_point_net/test/test.txt"
finished_files_dir = "D:\python project me\data\my_point_net/finished_files"

VOCAB_SIZE = 200000


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
    #with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_to_bin(input_file, out_file, makevocab=False):
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        # read the  input text file , make even line become article and odd line to be abstract（line number begin with 0）
        lines = read_text_file(input_file)
        for i, new_line in enumerate(lines):
            if i % 2 == 0:
                article = lines[i]
            if i % 2 != 0:
                abstract = "%s %s %s" % (SENTENCE_START, lines[i], SENTENCE_END)

                # Write to tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([bytes(article, encoding='utf-8')])
                # tf_example.features.feature['article'].bytes_list.value.extend([article])
                # tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
                tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract, encoding='utf-8')])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))

                # Write the vocab to file, if applicable
                if makevocab:
                    art_tokens = article.split(' ')
                    abs_tokens = abstract.split(' ')
                    abs_tokens = [t for t in abs_tokens if
                                  t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens]  # strip
                    tokens = [t for t in tokens if t != ""]  # remove empty
                    vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab.bin"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Read the text file, do a little postprocessing then write to bin files
    write_to_bin(test_file, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(val_file, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(train_file, os.path.join(finished_files_dir, "train.bin"), makevocab=True)