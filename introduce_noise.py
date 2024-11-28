import numpy as np
import secrets

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

def plaintext_encode(plaintext: str, m: int, noise_percentage: int) -> str:
    key_matrix = np.zeros((m, m)) # initialize empty key array

    # insert random values into key matrix
    # we need an invertible key matrix for decoding
    while np.linalg.det(key_matrix) == 0: # continue generating key matrices until an invertible one is found
        for i in range(m):
            for j in range(m):
                key_matrix[i, j] = secrets.randbelow(26)
    

    # convert plaintext into mx1 vector
    plaintext_matrix = []
    for c in plaintext:
        plaintext_matrix.append(letters_to_num[c])
    plaintext_matrix = np.atleast_2d(np.array(plaintext_matrix)).T # convert into mx1 numpy array


    # create ciphertext matrix
    ciphertext_matrix = np.matmul(key_matrix, plaintext_matrix)
    # convert values to valid range (0-25)
    ciphertext_matrix = ciphertext_matrix % 26

    # add noise with a given probability
    noise_matrix = np.zeros_like(ciphertext_matrix)
    for i in range(m):
      if secrets.randbelow(100) < noise_percentage: # n% chance of adding noise
                                                    # secrets generates a number 0-99 inclusive
                                                    # if it is less than noise_percentage, we apply noise to this cell
        noise_matrix[i, 0] = secrets.randbelow(26) # add random noise in range 0 to 25
    noisy_ciphertext_matrix = (ciphertext_matrix + noise_matrix) % 26

    # convert ciphertext matrix into ciphertext
    ciphertext = ""
    for k in range(m):
        ciphertext += num_to_letters[ciphertext_matrix[k, 0]]

    # convert noisy ciphertext matrix into noisy_ciphertext
    noisy_ciphertext = ""
    for k in range(m):
        noisy_ciphertext += num_to_letters[noisy_ciphertext_matrix[k, 0]]

    print("Key Matrix:\n", key_matrix)
    print("Plaintext Matrix:\n", plaintext_matrix)
    print("Ciphertext Matrix:\n", ciphertext_matrix)
    print("Noise Matrix:\n", noise_matrix)
    print("Noisy Ciphertext Matrix:\n", noisy_ciphertext_matrix)
    print("Ciphertext:", ciphertext)

    return noisy_ciphertext

if __name__=="__main__":
    # set parameters for testing
    m = 3 # key matrix dimensions
    plaintext = "CAT"

    print("Plaintext:", plaintext)
    noisy_ciphertext = plaintext_encode(plaintext, m, 50) # add noise parameter of 50
    print("Noisy Ciphertext:", noisy_ciphertext)
