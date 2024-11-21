#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#define FIFO1 "comm_pipe1"
#define FIFO2 "comm_pipe2"

void main() {
    char inp[300], rec[300];
    int inpSize, recSize, fd1, fd2;
    mknod(FIFO1, S_IFIFO | 0666, 0);
    mknod(FIFO2, S_IFIFO | 0666, 0);
    fd1 = open(FIFO1, O_WRONLY);
    fd2 = open(FIFO2, O_RDONLY);

    if (fd1 == -1 || fd2 == -1) {
        perror("Error opening FIFO");
        exit(1);
    }
    printf("Enter a message: ");
    fgets(inp, 300, stdin);
    inpSize = strlen(inp);
    inp[inpSize - 1] = '\0';
    if((inpSize = write(fd1, inp, inpSize)) == -1) {
        perror("Error writing to FIFO");
        exit(1);
    }

    printf("Sent %d bytes\n", inpSize);

    if((recSize = read(fd2, rec, 300)) == -1) {
        perror("Error reading from FIFO");
        exit(1);
    }
    rec[recSize] = '\0';
    printf("Received %d bytes: %s\n", recSize, rec);
    close(fd1);
    close(fd2);
    
}