#pragma once
#include<stdio.h>

typedef struct Node
{
    long long data;
    struct Node* next;
    struct Node* prev;
}D_Node;

// 함수 선언
void addFirstNode(D_Node** head, long long data);                           // 1번 --완료   12월 3일 확인
void addLastNode(D_Node** head, long long data);                            // 2번 --완료   12월 3일 확인
void addBeforeNode(D_Node* selectedNode, long long data);                   // 3번 --완료   12월 3일 확인
void addAfterNode(D_Node* selectedNode, long long data);                    // 4번 --완료   12월 3일 확인
void deleteFirstNode(D_Node** head);                                        // 5번 --완료   12월 3일 확인
void deleteLastNode(D_Node** head);                                         // 6번 --완료   12월 3일 확인
void deleteSearchNode(D_Node** head, int position);                         // 7번 --완료   12월 3일 확인
void deleteALLNode(D_Node** head);                                          // 8번 --완료   12월 3일 확인
void ModifyNode(D_Node* head);                                              // 9번 --완료   12월 3일 확인
void PrintSearchNode(D_Node* head, int position2);                          // 10번 --완료  12월 3일 확인
void printList(D_Node* head);                                               // 11번 --완료   12월 3일 확인
void LinearSearch(D_Node* head, int position4, int searchData);             // 12번 --완료   12월 3일 확인  // 노드 시작 1부터임 0아님
void MultipleLinearSearch(D_Node* head, long long searchData);              // 13번 --완료   12월 3일 확인  // 노드 시작 1
void binarySearch(D_Node* head, long long searchData);                      // 14번 --완료   12월 3일 확인  // 노드 시작 1 하나 숫자 찾는거
void multipleBinarySearch(D_Node* head, long long searchData);              // 15번  //문제 모든 값의 위치를 출력함
void bubbleSort(D_Node** head);                                             // 16번 --완료 확인
void insertionSort(D_Node** head);                                          // 17번 --완료 확인
void selectionSort(D_Node** head);
