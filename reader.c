#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define FIFO_NAME1 "comm_pipe1" 
#define FIFO_NAME2 "comm_pipe2" 

int main() {
    char s[300], vowel[20], send[300];
    int num, fd1, fd2, sig, k = 0, i, wordcnt = 0, charcnt = 0, linecnt = 0;

    mknod(FIFO_NAME1, S_IFIFO | 0666, 0);
    mknod(FIFO_NAME2, S_IFIFO | 0666, 0);

    printf("FIFO 2nd side connected, waiting for the other process to connect to the FIFO pipe...\n");

    fd1 = open(FIFO_NAME1, O_RDONLY); 
    fd2 = open(FIFO_NAME2, O_WRONLY);

    if (fd1 == -1 || fd2 == -1) {
        perror("Error opening FIFOs");
        exit(1);
    }

    printf("Established connection\n");

    if ((num = read(fd1, s, 300)) == -1) {
        perror("Error reading from FIFO1");
    } 
    else 
    {
        s[num] = '\0';  
        printf("Read %d bytes: \"%s\"\n", num, s);

        k = 0; wordcnt = 0; charcnt = 0; linecnt = 0;

        for(i = 0; i < num; i++) 
        {
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
        sprintf(send, "For the given sentence: Word count is %d\nVowel count is %d\nCharacter count is %d\nLines are %d\n", wordcnt, k, charcnt, linecnt);
        if ((sig = write(fd2, send, strlen(send))) == -1) 
        {
            perror("Error writing to FIFO2");
        } 
        else {
            printf("\nWritten successfully to FIFO2\n");
        }
    }

    close(fd1);
    close(fd2);

    return 0;
}   