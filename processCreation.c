#include <stdio.h>
#include <sys/types.h>
#include <ctype.h>
#include<stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>


int arr[100];
int n;
void ascending (int p, int n) {
  for(int i = 0; i < n - 1; i++) {
    int maxim = i;
    for (int j = 0; j < n; j++) {
      if(arr[i] > arr[j])
        maxim = j;

    }
    int temp = arr[maxim];
    arr[maxim] = arr[i];
    arr[i] = temp;
  }
  for (int i = 0; i < n ; i++) {
    sleep(1);
    printf("\t\t\t\ti=%d , p=%d , pid=%d , ppid=%d \n", arr[i], p, getpid(), getppid());
  }
}

void descending (int p, int n) {
  for(int i = 0; i < n - 1; i++) {
    int maxim = i;
    for (int j = 0; j < n; j++) {
      if(arr[i] < arr[j])
        maxim = j;

    }
    int temp = arr[maxim];
    arr[maxim] = arr[i];
    arr[i] = temp;
  }
  for (int i = 0; i < n ; i++) {
    sleep(10);
    printf("i=%d,p=%d,pid=%d,ppid=%d\n", arr[i], p, getpid(), getppid());

  }
}

void main() {
  printf("Enter the no of processes\n");
  int processes;
  scanf("%d", &processes);
  n = processes;
  for (int i = 0; i < processes; i++)
  {
    scanf("%d", &arr[i]);
  }
  int p = fork();
  if (p == 0)
  {
    ascending(p, n);
    exit(p);
  }
  else if (p > 0) {
    descending(p, n);
    exit(p);
  }
  else {
    printf("failed");
    exit(p);
  }
}


