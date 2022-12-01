import math


def sine_tone(stream, frequency, duration, volume=1, sample_rate=22050):
    n_samples = int(sample_rate * duration)



    s = lambda t: volume * math.sin(2 * math.pi * frequency * t / sample_rate)
    samples = (int(s(t) * 0x7f + 0x80) for t in range(n_samples))
    stream.write(bytes(bytearray(samples)))

    # fill remainder of frameset with silence
    # stream.write(b'\x80' * restframes)



