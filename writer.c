#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

// #include <stdio.h>: Provides functions for standard input/output (e.g., printf, fgets).
// #include <stdlib.h>: Provides functions like exit and memory-related utilities.
// #include <errno.h>: Used for handling system errors (e.g., perror prints error descriptions).
// #include <string.h>: Includes functions for string manipulation (e.g., strlen).
// #include <fcntl.h>: Provides file control options (e.g., open).
// #include <sys/types.h> and #include <sys/stat.h>: Required for file system and mknod (used to create FIFOs).
// #include <unistd.h>: Provides access to POSIX operating system API (e.g., read, write).

#define FIFO_NAME1 "comm_pipe1" 
#define FIFO_NAME2 "comm_pipe2" 

int main() {
    char input[300], received[300];
    int fd1, fd2, sig, num;

    mknod(FIFO_NAME1, S_IFIFO | 0666, 0); //leading zero is here indiactes that it is in octal format  not in decimal format 100
    mknod(FIFO_NAME2, S_IFIFO | 0666, 0);

    printf("FIFO 1st side connected, waiting for the other process to connect to the FIFO pipes...\n");

    // Open the FIFOs for writing and reading
    fd1 = open(FIFO_NAME1, O_WRONLY);    
    fd2 = open(FIFO_NAME2, O_RDONLY);

    if (fd1 == -1 || fd2 == -1) {
        perror("Error opening FIFOs");
        exit(1);
    }

    printf("Established connection\n");

    printf("Enter a string to send to the reader process: ");
    fgets(input, sizeof(input), stdin);
    input[strlen(input) - 1] = '\0'; 

    if ((sig = write(fd1, input, strlen(input))) == -1) {
        perror("Error writing to FIFO1");
    } 
    else 
    {
        printf("\nSent data to FIFO1\n");
    }

    if ((num = read(fd2, received, 300)) == -1) {
        perror("Error reading from FIFO2");
    } else {
        received[num] = '\0'; 
        printf("\nReceived the following from the reader process:\n%s\n", received);
    }

    // Close the FIFOs
    close(fd1);
    close(fd2);

    return 0;
}