#include <stdio.h> // Include standard input/output header file

int main() { // Main function
    int processIndex, processCount, time, processCompleted = 0, timeQuantum, remainingProcesses; // Declare variables
    int totalTurnaroundTime = 0, totalWaitingTime = 0; // Initialize total turnaround and waiting time

    printf("Enter the total no. of process"); // Prompt user for total number of processes
    scanf("%d", &processCount); // Read total number of processes
    remainingProcesses = processCount; // Initialize remaining processes
    printf("\nEnter the time quantum"); // Prompt user for time quantum
    scanf("%d", &timeQuantum); // Read time quantum
    int burstTime[processCount], arrivalTime[processCount], remainingTime[processCount]; // Declare arrays for burst time, arrival time, and remaining time
    for (int i = 0; i < processCount; i++) // Loop through each process
    {
        printf("Enter the arrival Time, burst time of process %d\t", i + 1); // Prompt user for arrival and burst time
        scanf("%d %d", &arrivalTime[i], &burstTime[i]); // Read arrival and burst time
        remainingTime[i] = burstTime[i]; // Initialize remaining time
    }

    printf("Starting the Round Robin\n"); // Indicate the start of Round Robin scheduling

    for (time = 0, processIndex = 0; remainingProcesses != 0;) // Loop until all processes are completed
    {
        if (remainingTime[processIndex] > 0 && remainingTime[processIndex] <= timeQuantum) // If process can be completed in current time quantum
        {
            time += remainingTime[processIndex]; // Increment time
            remainingTime[processIndex] = 0; // Process is completed
            processCompleted = 1; // Set process completed flag
        }
        else if(remainingTime[processIndex] > timeQuantum) { // If process cannot be completed in current time quantum
            time += timeQuantum; // Increment time by time quantum
            remainingTime[processIndex] -= timeQuantum; // Decrease remaining time
        }

        if (processCompleted == 1 && remainingTime[processIndex] == 0) // If process is completed
        {
            remainingProcesses--; // Decrease remaining processes
            printf("P[%d]\t|\t%d\t|\t%d\n", processIndex + 1, time - arrivalTime[processIndex],
            time - arrivalTime[processIndex] - burstTime[processIndex]); // Print process details
            totalWaitingTime += time - arrivalTime[processIndex] - burstTime[processIndex]; // Update total waiting time
            totalTurnaroundTime += time - arrivalTime[processIndex]; // Update total turnaround time
            processCompleted = 0; // Reset process completed flag
        }

        if (processIndex == processCount - 1) // If last process
        {
            processIndex = 0; // Reset process index
        }
        else if (arrivalTime[processIndex + 1] <= time) // If next process has arrived
        {
            processIndex++; // Move to next process
        }
        else
            processIndex = 0; // Reset process index
    }
    printf("average wait time %f", (float) totalWaitingTime / processCount); // Print average waiting time
    printf("average turnaround time %f", (float) totalTurnaroundTime / processCount); // Print average turnaround time
}