import numpy as np
import scipy
import itertools as it
from introduce_noise import noise_encode

plaintext = "LINALG"
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

def noise_decode(noisy_ciphertext: str, noisy_ciphertext_matrix: int, m: int, w: str) -> str:
    key_matrices = it.product(range(26), repeat=4) # generate every possible key matrix in array format
    
    for possible_key in key_matrices:
        possible_matrix = np.reshape(possible_key, (m, m)) # reshape the array into m*m matrix format
        if np.linalg.det(possible_matrix) == 0: # only consider invertible matrices
            continue
        else:
            possible_plaintext_matrix = np.matmul(possible_matrix, noisy_ciphertext_matrix)
            possible_plaintext_matrix = possible_plaintext_matrix % 26

            decoded_text = ""
            for k in range(m):
                decoded_text += num_to_letters[possible_plaintext_matrix[k, 0]]
            if decoded_text == w:
                return possible_matrix
    
    return -1


if __name__=="__main__":
    # set parameters for testing
    m = 2 # key matrix dimensions
    full_ciphertext = ""

    segments = [plaintext[i:i+2] for i in range(0, len(plaintext), 2)]
    for w in segments:
        print("Plaintext:", w)
        noisy_ciphertext, noisy_ciphertext_matrix = noise_encode(w, m, 0)
        full_ciphertext += noisy_ciphertext
        print("Noisy Ciphertext:", noisy_ciphertext)
        decoded_matrix = noise_decode(noisy_ciphertext, noisy_ciphertext_matrix, m, w)
        print("Decoded Matrix:\n", decoded_matrix)

    print("Full Ciphertext:", full_ciphertext)


    """
    for n in range(0, 101, 10):
        success_count = 0
        for i in range(3):
            noisy_ciphertext, noisy_ciphertext_matrix = noise_encode(plaintext, m, n) # add noise parameter of 50
            print("Noisy Ciphertext:", noisy_ciphertext)
            decoded_matrix = noise_decode(noisy_ciphertext, noisy_ciphertext_matrix, m)
            print("Decoded Matrix:\n", decoded_matrix)
            if type(decoded_matrix) != int:
                success_count += 1
        results[n] = success_count
    
    print(results)
    """