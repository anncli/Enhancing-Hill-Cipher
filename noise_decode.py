import numpy as np
import Levenshtein
import itertools as it
from introduce_noise import noise_encode

# CONSTANTS FOR TESTING
SIMILARITY_THRESHOLD = 0.7
DIM = 2 # key matrix dimensions
plaintext = "HEYGUESSWHATLINALGISSOCOOL"

letters_to_num = dict({
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
})
num_to_letters = dict({
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
})

def compare_str(decoded_text: str, plaintext: str, threshold=SIMILARITY_THRESHOLD):
    distance = Levenshtein.distance(decoded_text, plaintext)
    similarity_ratio = 1 - distance / len(decoded_text) 
    return similarity_ratio >= threshold

decoded_text = ""
def noise_decode(noisy_ciphertext: str, m: int) -> str:
    key_matrices = it.product(range(-25, 26), repeat=4) # generate every possible key matrix in array format
    possible_matrices = set()

    segments = []
    for i in range(0, len(noisy_ciphertext), 2):
        segments.append(noisy_ciphertext[i:i+2])

    for possible_key in key_matrices:
        possible_matrix = np.reshape(possible_key, (m, m)) # reshape the array into m*m matrix format
        if np.linalg.det(possible_matrix) == 0: # only consider invertible matrices
            continue

        decoded_text = ""
        for segment in segments:
            ciphertext_matrix = np.array([letters_to_num[c] for c in segment]).reshape((m, 1))

            possible_plaintext_matrix = np.matmul(possible_matrix, ciphertext_matrix)
            possible_plaintext_matrix = possible_plaintext_matrix % 26

            for k in range(m):
                decoded_text += num_to_letters[possible_plaintext_matrix[k, 0]]
        
        if compare_str(decoded_text, plaintext):
            print("Decoded Text:", decoded_text)
            possible_matrices.add(tuple(map(tuple, possible_matrix)))
        
    return possible_matrices


if __name__=="__main__":
    segments = []
    for i in range(0, len(plaintext), 2):
        segments.append(plaintext[i:i+2])
    
    full_ciphertext = ""
    for w in segments:
        print("Plaintext:", w)
        noisy_ciphertext, noisy_ciphertext_matrix = noise_encode(w, DIM, 18)
        full_ciphertext += noisy_ciphertext
        print("Noisy Ciphertext:", noisy_ciphertext)
    
    print("Full Ciphertext:", full_ciphertext)

    decoded_matrix = noise_decode(full_ciphertext, DIM)
    print("Solution Key Matrix:\n", decoded_matrix)