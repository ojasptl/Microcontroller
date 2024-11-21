#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>


struct fcfc {
    char name[100];
    int arrivalTime;
    int burstTime;
    int tat;
    int waiting;
};

int main() { 
    printf("enter the total processes\n");
    int processes, totalWait = 0, totalTAT = 0;
    scanf("%d", &processes);
    struct fcfc arr[processes];
    for (int i = 0; i < processes; i++) {
        printf("Enter the name, arrival time and bursttime\t");
        scanf("%s %d %d", arr[i].name, &arr[i].arrivalTime, &arr[i].burstTime);
    }
    for (int i = 0; i < processes - 1; i++) {
        for (int j = i + 1; j < processes; j++) {
            if (arr[i].arrivalTime > arr[j].arrivalTime) {
                struct fcfc temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
    printf("Printing the Sorted array\n");
    for (int i = 0; i < processes; i++) {
        printf("Process: %s, Arrival Time: %d, Burst Time: %d \n", arr[i].name, arr[i].arrivalTime, arr[i].burstTime);
    }

    arr[0].tat = arr[0].arrivalTime + arr[0].burstTime;
    for (int i = 1; i < processes; i++) {
        if (arr[i - 1].tat >= arr[i].arrivalTime) {
            arr[i].tat = arr[i - 1].tat + arr[i].burstTime;
        }
        else {
            arr[i].tat = arr[i].arrivalTime + arr[i].burstTime;
        }
    }

    for (int i = 0; i < processes; i++) {
        arr[i].tat = arr[i].tat - arr[i].arrivalTime;
        arr[i].waiting = arr[i].tat - arr[i].burstTime;
        totalTAT += arr[i].tat;
        totalWait += arr[i].waiting;
    }

    printf("The Waiting time is %f", (float) totalWait / processes);
    printf("the turnaround time is %f", (float) totalTAT / processes);
    return 0;
}