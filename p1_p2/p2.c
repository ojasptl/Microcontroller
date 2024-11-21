#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define FIFO1 "comm_pipe1"
#define FIFO2 "comm_pipe2"

void main() { 
    char inp[300], rec[300];
    int inpsize, recsize, fd1, fd2, linecnt = 0, wrdcnt = 0, charcnt = 0, k = 0;
    char vowels[100];
    vowels[0] = '\0';

    mknod(FIFO1, S_IFIFO | 0666, 0);
    mknod(FIFO2, S_IFIFO | 0666, 0);

    printf("Waiting for reader to connect...\n");
    fd1 = open(FIFO1, O_RDONLY);
    fd2 = open(FIFO2, O_WRONLY);

    if (fd1 == -1 || fd2 == -1) {
        perror("Error opening FIFO");
        exit(1);
    }

    if((inpsize = read(fd1, inp, 300)) == -1) {
        perror("Error reading from FIFO");
        exit(1);
    }

    printf("Received %d bytes: %s\n", inpsize, inp);

    for (int i = 0; i < inpsize; i++) {
                if(s[i] == '.') 
                {
                    linecnt++; 
                }

                if (s[i] == ' ' || s[i] == '.' || s[i] == '\0') 
                {
                    if (i > 0 && s[i - 1] != ' ' && s[i - 1] != '\n') {
                        wordcnt++; 
                    }
                } 
                else 
                {
                    charcnt++;  
                    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u') {
                        vowel[k++] = s[i];  
                    }
                }
        }

        if (num > 0 && s[num - 1] != ' ' && s[num - 1] != '\n') {
            wordcnt++;
        }
        vowel[k] = '\0';
    sprintf(rec, "Vowels: %s\nCharacters: %d\nWords: %d\nLines: %d\n", vowels, charcnt, wrdcnt, linecnt);
    recsize = strlen(rec);
    rec[recsize - 1] = '\0';

    if((recsize = write(fd2, rec, recsize)) == -1) {
        perror("Error writing to FIFO");
        exit(1);
    }

    printf("Sent %d bytes\n", recsize);
    close(fd1);
    close(fd2);
}   