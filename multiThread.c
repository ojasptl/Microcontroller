#include<stdio.h>
#include<stdlib.h>
#include<semaphore.h>
#include<pthread.h>
#include<unistd.h>

sem_t sem;
pthread_mutex_t x;
int readcount = 0;
int s = 5;

void * writer (void * a) {
  int i = 0;
  while (i < 3) {
    sem_wait(&sem);
    s += 5;
    printf("Process %d is writing %d\n", i, s);
    sem_post(&sem);
    sleep(rand() % 10);
    i++;
  }
  return NULL;
}

void * reader (void * a) {
  int i= 0;
  while ( i < 3) {
    pthread_mutex_lock(&x);
    readcount++;    
    if ( readcount == 1) {
      sem_wait(&sem);
    }
    pthread_mutex_unlock(&x);
    printf("\t\tReader %d is reading %d\n", i, s);
    pthread_mutex_lock(&x);
    readcount--;
    if (readcount == 0) {
      sem_post(&sem);
    }
    pthread_mutex_unlock(&x);
    sleep(rand() % 10);
    i++;

  }
  return NULL;
}

void main() {
  printf("There will be multithreading process going on now");
  int op;
  pthread_mutex_init(&x, NULL);
  sem_init(&sem, 0, 1);
  pthread_t read[5], write[5];

  do {
    printf("Enter your choice\n1 for giving reader priority\n2 for exit");
    scanf("%d", &op);
    if (op == 2) {
      exit(1);
    }
    else {
      for (int i = 0; i < 5; i++) {
        pthread_create(&read[i], NULL, reader, (void *)(long long) i);
      }
      
      for (int i = 0; i < 5; i++) {
        pthread_create(&write[i], NULL, writer, (void *)(long long) i);
      }

      for(int i=0;i<3;i++){
        pthread_join(read[i],NULL);
      }
      
      for(int i=0;i<3;i++){
          pthread_join(write[i],NULL);
      }
    } 
  }
  while (op != 2);

}
