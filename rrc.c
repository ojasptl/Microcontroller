#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int main() {
    printf("Enter the no of the processes\n");
    int remainingProcesses, status = 0, processIndex, processCount, time = 0, timeQuantum;
    int totalWait = 0, totalTAT = 0;

    scanf("%d", &processCount);
    remainingProcesses = processCount;
    printf("Enter the time quantum\t");
    scanf("%d", &timeQuantum);
    int burstTime[processCount], arrivalTime[processCount], remainingTime[processCount];
    for (int i = 0; i < processCount; i++) {
        printf("Enter the arrival time and burst time in ascending order for %d\n", i + 1);
        scanf("%d %d", &arrivalTime[i], &burstTime[i]);
        remainingTime[i] = burstTime[i];
    }
    for (processIndex = 0; remainingProcesses > 0;) {
        if (remainingTime[processIndex] <= timeQuantum && remainingTime[processIndex] > 0) {
            time += remainingTime[processIndex];
            remainingTime[processIndex] = 0;
            status = 1;
        }
        else if (remainingTime[processIndex] > timeQuantum) {
            time += timeQuantum;
            remainingTime[processIndex] -= timeQuantum;
        }

        if (status == 1 && remainingTime[processIndex] == 0) {
            printf("The TAT is %d and the Waiting Time is %d\n", time - arrivalTime[processIndex], time  - arrivalTime[processIndex] - burstTime[processIndex]);
            totalTAT += time - arrivalTime[processIndex];
            totalWait += time  - arrivalTime[processIndex] - burstTime[processIndex];
            remainingProcesses--;
            status = 0;
        }

        if (processIndex == processCount - 1) {
            processIndex = 0;
        }
        else if (arrivalTime[processIndex + 1] <= time) {
            processIndex++;
        }
        else {
            processIndex = 0;
        }
    }
    printf("The avg waiting time is %f\n", (float)totalWait / processCount);
    printf("The avg tat is %f\n", (float)totalTAT / processCount);
    return 0;

}