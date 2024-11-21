#include <stdio.h>
#include <sys/types.h>
#include <ctype.h>
#include<stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int arr[100];
int n;
void ascending (int p, int n) {
  for(int i = 0; i < n - 1; i++) {
    int maxim = i;
    for (int j = 0; j < n; j++) {
      if(arr[i] > arr[j])
        maxim = i;

    }
  }
  for (int i = 0; i < n ; i++) {
    printf("")
  }
}
