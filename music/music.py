from music21 import converter, note, chord
from music21 import converter, note, chord, pitch, stream, instrument

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def get_notes_from_midi(file):
    """ 从MIDI文件中提取音符和和弦 """
    try:
        midi = converter.parse(file)
        notes_to_parse = midi.flat.notes
        notes = []
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        return notes
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
# # 修改后的函数，接受文件列表
# def get_notes_from_midis(file_paths):
#     notes = []
#     for file_path in file_paths:
#         midi = converter.parse(file_path)
#         notes_to_parse = None
#
#         parts = instrument.partitionByInstrument(midi)
#         if parts:  # file has instrument parts
#             notes_to_parse = parts.parts[0].recurse()
#         else:  # file has notes in a flat structure
#             notes_to_parse = midi.flat.notes
#
#         for element in notes_to_parse:
#             if isinstance(element, note.Note):
#                 notes.append(str(element.pitch))
#             elif isinstance(element, chord.Chord):
#                 notes.append('.'.join(str(n) for n in element.normalOrder))
#     return notes

#加载一个MIDI文件并提取音符
# 使用绝对路径替换以下路径
midi_file = 'C:\\Users\\Administrator\\Desktop\\myhart.mid'
notes = get_notes_from_midi(midi_file)
if notes:
    print(notes)
else:
    print("No notes were extracted from the MIDI file.")

sequence_length = 100
n_vocab = len(set(notes))
note_to_int = dict((note, number) for number, note in enumerate(set(notes)))
# 创建音符到整数和整数到音符的映射
note_to_int = dict((note, number) for number, note in enumerate(set(notes)))
int_to_note = dict((number, note) for note, number in note_to_int.items())

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)
network_output = to_categorical(network_output)

# 创建LSTM模型
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(n_vocab, activation='softmax'))  # 直接在 Dense 层中指定激活函数
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# 训练模型
model.fit(network_input, network_output, epochs=100, batch_size=64)

import random
from music21 import instrument, note, stream, chord


# 生成音乐片段
def generate_notes(model, network_input, int_to_note, n_vocab, num_generate=500):
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(num_generate):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


# 将生成的音符转换为MIDI文件
def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(pitch.Pitch(int(current_note)))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')


# 执行生成并保存MIDI文件
generated_notes = generate_notes(model, network_input, int_to_note, n_vocab)
create_midi(generated_notes)

