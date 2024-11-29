import numpy as np
import secrets
import sympy as sp
import time
from memory_profiler import memory_usage

letters_to_num = dict({
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
})

num_to_letters = {v: k for k, v in letters_to_num.items()}  # reverse mapping 

# method 1 - introducing noise
def plaintext_encode_method1(plaintext: str, m: int, noise_percentage: int) -> str:
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

    return noisy_ciphertext

# method 2 - hybrid encryption
def plaintext_encode_method2(plaintext: str, m: int) -> str:
    key_matrix = np.zeros((m, m)) # initialize empty key array

    # insert random values into key matrix
    # we need an invertible key matrix for decoding
    while np.linalg.det(key_matrix) == 0: # continue generating key matrices until an invertible one is found
        for i in range(m):
            for j in range(m):
                key_matrix[i, j] = secrets.randbelow(26)
    

    # encrypt key matrix with RSA encryption
    p = sp.randprime(0, 1000); # generate large prime p
    q = sp.randprime(0, 1000); # generate large prime q
    n = p * q # n is part of public and private keys
    phi = (p - 1) * (q - 1) # totient: number of integers less than n coprime to n

    # choose e, a value between 1 and phi(n) coprime to phi(n)
    e = secrets.randbelow(phi)
    while sp.gcd(e, phi) != 1:
        e = secrets.randbelow(phi)

    # d is private key, modular multiplicative inverse of e modulo phi(n)
    d = sp.mod_inverse(e, phi)

    flatten_key_RSA = key_matrix.flatten()

    encrypted_key_RSA = []
    for value in flatten_key_RSA:
        RSA_value = pow(int(value), e, n) # = m^n % e
        encrypted_key_RSA.append(RSA_value)

    RSA_key_matrix = np.array(encrypted_key_RSA).reshape(m, m)

    # for testing, decrypt encrypted RSA matrix to ensure it matches original
    decrypted_key_RSA = []
    for value in encrypted_key_RSA:
        decrypted_value = pow(int(value), d, n) # = c^d % n
        decrypted_key_RSA.append(decrypted_value)
    decrypted_key_RSA = np.array(decrypted_key_RSA).reshape(m, m)

    # convert plaintext into mx1 vector
    plaintext_matrix = []
    for c in plaintext:
        plaintext_matrix.append(letters_to_num[c])
    plaintext_matrix = np.atleast_2d(np.array(plaintext_matrix)).T # convert into mx1 numpy array

    # create ciphertext matrix
    ciphertext_matrix = np.matmul(key_matrix, plaintext_matrix)
    # convert values to valid range (0-25)
    ciphertext_matrix = ciphertext_matrix % 26

    # convert ciphertext matrix into ciphertext
    ciphertext = ""
    for k in range(m):
        ciphertext += num_to_letters[ciphertext_matrix[k, 0]]

    return ciphertext

# time and space complexity testing
def test_performance(plaintext, method, m, *args):
    start_time = time.time()
    mem_usage = memory_usage((method, (plaintext, m, *args)), interval=0.01, retval=True)
    end_time = time.time()
    execution_time = end_time - start_time
    memory_used = max(mem_usage[0]) - min(mem_usage[0])
    return execution_time, memory_used, mem_usage[1]

if __name__=="__main__":
    # set parameters for testing
    m = 3 # key matrix dimensions
    plaintext = "CAT"

    # testing for method 1
    print("Testing Method 2.2:")
    exec_time_2_2, mem_2_2, result_2_2 = test_performance(
        plaintext, plaintext_encode_method1, m, 50)
    print(f"Execution Time: {exec_time_2_2:.6f}s")
    print(f"Memory Used: {mem_2_2:.6f} MiB")
    print(f"Resulting Ciphertext: {result_2_2}")

    # testing for method 2
    print("\nTesting Method 2.3:")
    exec_time_2_3, mem_2_3, result_2_3 = test_performance(
        plaintext, plaintext_encode_method2, m)
    print(f"Execution Time: {exec_time_2_3:.6f}s")
    print(f"Memory Used: {mem_2_3:.6f} MiB")
    print(f"Resulting Ciphertext: {result_2_3}")