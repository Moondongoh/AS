#include<stdlib.h>
#include "LinkedList.h"

/*
* 
*/

//1번 앞 노드 데이터 추가
void addFirstNode(D_Node** head, long long data)
{
    D_Node* newNode = (D_Node*)malloc(sizeof(D_Node));  // 새로운 노드 동적 할당

    newNode->data = data;
    newNode->next = *head;                                             // 현재 헤드를 새로운 노드의 다음으로 설정
    *head = newNode;                                                   // 헤드를 새로운 노드로 업데이트
}

/*
*
*/

// 2번 뒤 노드 데이터 추가
void addLastNode(D_Node** head, long long data) {
    // 새로운 노드 동적 할당
    D_Node* newNode = (D_Node*)malloc(sizeof(D_Node));
    newNode->data = data;
    newNode->next = NULL;

    if (*head == NULL) {
        // 연결 리스트가 비어있는 경우
        *head = newNode;
    }
    else {
        // 연결 리스트의 끝까지 이동
        struct Node* cur = *head;
        while (cur->next != NULL) {
            cur = cur->next;
        }

        // 새로운 노드를 연결 리스트의 끝에 추가
        cur->next = newNode;
    }
}

/*
*
*/

// 3번 기준 노드 앞 데이터 추가
void addBeforeNode(D_Node** head, long long searchData, long long data) {
    D_Node* cur = *head;
    D_Node* prev = NULL;

    // 노드의 데이터가 searchData와 일치하는 노드를 찾음
    while (cur != NULL && cur->data != searchData) {
        prev = cur;
        cur = cur->next;
    }

    if (cur == NULL) {
        // searchData와 일치하는 노드를 찾지 못한 경우
        printf("일치하는 노드를 찾을 수 없습니다.\n");
    }
    else {
        // 새로운 노드 동적 할당
        D_Node* newNode = (D_Node*)malloc(sizeof(D_Node));
        if (newNode == NULL) {
            printf("메모리 할당 오류\n");
            exit(EXIT_FAILURE);
        }

        newNode->data = data;

        // 이전 노드가 없는 경우 (즉, 찾은 노드가 헤드인 경우)
        if (prev == NULL) {
            newNode->next = *head;
            *head = newNode; // 헤드를 새로운 노드로 업데이트
        }
        else {
            // 이전 노드가 있는 경우
            newNode->next = prev->next;
            prev->next = newNode;
        }
    }
}

/*
*
*/

// 4번 기준 노드 뒤 데이터 추가
void addAfterNode(D_Node* head, long long searchData, long long data) {
    D_Node* cur = head;

    // 노드의 데이터가 searchData와 일치하는 노드를 찾음
    while (cur != NULL && cur->data != searchData) {
        cur = cur->next;
    }

    if (cur == NULL) {
        // searchData와 일치하는 노드를 찾지 못한 경우
        printf("일치하는 노드를 찾을 수 없습니다.\n");
    }
    else {
        // 새로운 노드 동적 할당
        D_Node* newNode = (D_Node*)malloc(sizeof(D_Node));
        if (newNode == NULL) {
            printf("메모리 할당 오류\n");
            exit(EXIT_FAILURE);
        }

        newNode->data = data;

        // 새로운 노드를 선택된 노드 뒤에 추가
        newNode->next = cur->next;
        cur->next = newNode;

    }
}

/*
*
*/

// 5번 앞 노드 데이터 삭제
void deleteFirstNode(D_Node** head) {
    if (*head == NULL) {
        printf("삭제할 노드가 없습니다.\n");
        return;
    }

    D_Node* newHead = (*head)->next;
    free(*head);
    *head = newHead;
}

/*
*
*/

// 6번 뒤 노드 데이터 삭제
void deleteLastNode(D_Node** head)
{
    if (*head == NULL)
    {
        printf("삭제할 노드가 없습니다.\n");
        return;
    }

    if ((*head)->next == NULL)
    {
        free(*head);
        *head = NULL;
    }
    else
    {
        D_Node* cur = *head;
        while (cur->next->next != NULL)
        {
            cur = cur->next;
        }
        free(cur->next);
        cur->next = NULL;
    }
}

// 7번 임의 위치 데이터 삭제
void deleteSearchNode(D_Node** head, int position)
{
    if (*head == NULL)
    {
        printf("삭제할 노드가 없습니다.\n");
        return;
    }

    if (position == 0)
    {
        // 헤드 노드를 삭제하는 경우
        D_Node* temp = *head;
        *head = (*head)->next;
        free(temp);
        return;
    }

    // 찾을 위치까지 이동
    D_Node* cur = *head;
    D_Node* prev = NULL;
    int count = 0;

    while (cur != NULL && count < position)
    {
        prev = cur;
        cur = cur->next;
        count++;
    }

    if (cur == NULL)
    {
        // 찾을 위치가 리스트의 길이보다 큰 경우
        printf("삭제할 위치를 찾을 수 없습니다.\n");
        return;
    }

    // 찾은 노드를 삭제
    prev->next = cur->next;
    free(cur);
}

/*
*
*/

// 8번 모든 노드 데이터 삭제
void deleteALLNode(D_Node** head)
{
    D_Node* prev = *head;

    while (*head)
    {
        *head = (*head)->next;

        //printf("삭제할 데이터 값 : %d\n", prev->data);
        free(prev);
        prev = *head;
    }
}

/*
*
*/

// 9번 임의 노드 데이터 수정 //수정 완료
void ModifyNode(D_Node* head)
{
    int position3 = 0;
    D_Node* cur = head;

    // 노드의 데이터가 position3와 일치하는 노드를 찾음
    int count = 0;
    while (cur != NULL && count < position3)
    {
        cur = cur->next;
        count++;
    }

    if (cur == NULL)
    {
        // position3와 일치하는 노드를 찾지 못한 경우
        printf("일치하는 노드를 찾을 수 없습니다.\n");
    }
    else
    {
        int newData;
        printf("수정할 새로운 데이터를 입력하세요: ");
        scanf("%d", &newData);

        cur->data = newData;
        printf("데이터가 성공적으로 수정되었습니다.\n");
    }
}


// 10번 임의 노드 데이터 값 읽기
void PrintSearchNode(D_Node* head, int position2)
{
    int count = 0;
    while (head != NULL && count < position2)
    {
        head = head->next;
        count++;
    }

    if (head != NULL)
    {
        // 특정 위치의 노드 값을 출력
        printf("특정 위치의 노드 값: %d\n", head->data);
    }
    else
    {
        // 특정 위치를 찾지 못한 경우 처리
        printf("특정 위치를 찾을 수 없습니다.\n");
    }
}

/*
*
*/

// 11번 순회
void printList(D_Node* head)
{
    int min = INT_MIN; // long long LLONG 수정
    int max = INT_MAX;
    // cur을 head로
    struct Node* cur = head;
    /*
    현재 노드가 NULL이 아닐때까지 why?
    연결리스트의 끝은 NULL이기에 NULL이다? 종료 ㄱ
    */
    while (cur != NULL)
    {
        printf("%d ", cur->data);
        cur = cur->next;
    }
    printf("\n");
    printf("MIN : %d\n", min);
    printf("MAX : %d\n", max);
}

/*
*
*/

// 12번 단일 선형 탐색?
void LinearSearch(D_Node* head, int position4, int searchData)
{

    int count = 0;

    //printf("찾을 값을 입력하세요: ");
    //if (scanf("%d", &searchdata) != 1)
    //{
    //    printf("잘못된 값을 입력 했습니다\n");
    //    while (getchar() != '\n');
    //}

    D_Node* cur = head;

    // 노드의 데이터가 searchData와 일치하는 노드를 찾음
    while (cur != NULL && cur->data != searchData && count < position4)
    {
        cur = cur->next;
        count++;
    }

    if (cur == NULL)
    {
        // searchData와 일치하는 노드를 찾지 못한 경우
        printf("일치하는 노드를 찾을 수 없습니다.\n");
    }
    else
    {
        printf("%d 값의 노드 위치는: %d\n", searchData, count + 1);
    }
}

/*
*
*/

// 13번 다중 선형 탐색
void MultipleLinearSearch(D_Node* head, int searchData)
{
    int position = 0;
    int count = 0;


    D_Node* cur = head;

    // 노드의 데이터가 searchData와 일치하는 노드를 찾음
    while (cur != NULL) {
        if (cur->data == searchData) {
            printf("%d 값의 노드 위치는: %d\n", searchData, count + 1);
        }
        cur = cur->next;
        count++;
    }

    if (count == 0) {
        // 일치하는 노드를 찾지 못한 경우
        printf("일치하는 노드를 찾을 수 없습니다.\n");
    }
}

/*
*
*/

// 14번 단일 이진 탐색
void binarySearch(D_Node* head, int searchData) {
    int count = 0;  // 노드 수 셀 변수
    int position = -1;

    // 리스트의 노드 수 계산
    D_Node* temp = head;
    while (temp != NULL) {
        count++;
        temp = temp->next;
    }

    // 노드의 데이터가 searchData와 일치하는 노드를 찾음
    int low = 0;
    int high = count - 1;

    while (low <= high) {
        int mid = (low + high) / 2;

        D_Node* cur = head;
        for (int i = 0; i < mid; i++) {
            cur = cur->next;
        }

        if (cur->data == searchData) {
            position = mid;
            break;
        }
        else if (cur->data < searchData) {
            low = mid + 1;
        }
        else {
            high = mid - 1;
        }
    }
    if (position != -1) {
        printf("%d 값의 노드 위치는: %d\n", searchData, position + 1);
    }
    else {
        // searchData와 일치하는 노드를 찾지 못한 경우
        printf("일치하는 노드를 찾을 수 없습니다.\n");
    }
}

/*
* 이게 제일 문제
*/

// 15번 다중 이진 탐색
void multipleBinarySearch(D_Node* head, int searchData) {
    int position = -1;
    int count = 0;  // 노드 수 셀 변수

    // 리스트의 노드 수 계산
    D_Node* temp = head;
    while (temp != NULL) {
        count++;
        temp = temp->next;
    }

    int low = 0;
    int high = count - 1;

    while (low <= high) {
        int mid = (low + high) / 2;
        int data;

        D_Node* cur = head;
        for (int i = 0; i < mid; i++) {
            cur = cur->next;
        }

        if (cur->data == searchData) {
            position = mid;
            printf("%d 값의 노드 위치는: %d\n", searchData, position+1);
            ///////////////////////////////////////
            // 왼쪽으로 이동하며 출력
            int left = position - 1;
            D_Node* leftNode = head;
            while (left >= 0 && leftNode->data == searchData) {
                printf("%d 값의 노드 위치는: %d\n", searchData, left + 1);
                left--;
                leftNode = leftNode->next;
            }

            // 오른쪽으로 이동하며 출력
            int right = position + 1;
            D_Node* rightNode = cur->next; // 현재 위치에서의 다음 노드부터 시작
            while (right < count && rightNode->data == searchData) {
                printf("%d 값의 노드 위치는: %d\n", searchData, right + 1);
                right++;
                rightNode = rightNode->next;
            }

            return;  // 찾은 후에는 더 이상 진행하지 않음
        }

        else if (cur->data < searchData) {
            low = mid + 1;
        }
        else {
            high = mid - 1;
        }
    }

    // searchData와 일치하는 노드를 찾지 못한 경우
    printf("일치하는 노드를 찾을 수 없습니다.\n");
}

/*
* 
*/

// 16번 버블 정렬
void bubbleSort(D_Node** head)
{
    int swapped;
    D_Node* cur;
    D_Node* last = NULL;

    // 빈 리스트 또는 하나의 노드만 있는 경우 정렬할 필요 없음
    if (*head == NULL || (*head)->next == NULL) {
        return;
    }

    do {
        swapped = 0;
        cur = *head;

        while (cur->next != last) {
            // 현재 노드와 다음 노드를 비교하여 교환
            if (cur->data > cur->next->data) {
                int temp = cur->data;
                cur->data = cur->next->data;
                cur->next->data = temp;
                swapped = 1;
            }

            cur = cur->next;
        }

        last = cur;

    } while (swapped);
}

/*
* 
*/

// 정렬된 연결리스트에 새로운 값을 삽입함.
void sortedInsert(D_Node** head, int data)
{
    D_Node* newNode = (D_Node*)malloc(sizeof(D_Node));
    newNode->data = data;
    newNode->next = NULL;

    // 만약 리스트가 비어있거나 새로운 노드의 데이터가 헤드의 데이터보다 작다면
    // 헤드에 삽입
    if (*head == NULL || (*head)->data >= newNode->data) {
        newNode->next = *head;
        *head = newNode;
    }
    else {
        // 그렇지 않으면 올바른 위치를 찾아 삽입
        D_Node* cur = *head;
        while (cur->next != NULL && cur->next->data < newNode->data) {
            cur = cur->next;
        }
        newNode->next = cur->next;
        cur->next = newNode;
    }
}

/*
* 
*/

// 17번 삽입 정렬
void insertionSort(D_Node** head)
{
    D_Node* sorted = NULL; // 정렬된 리스트의 헤드

    while (*head != NULL) {
        D_Node* cur = *head;
        *head = (*head)->next;

        sortedInsert(&sorted, cur->data);
    }

    // 정렬된 리스트를 원래 리스트로 복사
    *head = sorted;
}

/*
* cur이라는 현재 노드 가르키는 포인터 초기화
* 임시 노드
* 현재 노드가 끝이 되기전까지 ㄱ
* 현재 값을 최솟값으로 인식
* 왕게임 마냥 스타트하면 될듯?
* while문으로 현재 노드 보다 작으면 갱신
* 다음 노드로 이동
*/

// 18번 선택 정렬
void selectionSort(D_Node** head)
{
    D_Node* cur = *head;
    D_Node* temp = NULL;

    while (cur != NULL) {
        D_Node* minNode = cur;
        D_Node* searchNode = cur->next;

        // 최솟값 찾기
        while (searchNode != NULL) {
            if (searchNode->data < minNode->data) {
                minNode = searchNode;
            }
            searchNode = searchNode->next;
        }

        // 최솟값과 현재 노드의 데이터 교환
        if (minNode != cur) {
            int tempData = cur->data;
            cur->data = minNode->data;
            minNode->data = tempData;
        }

        cur = cur->next;
    }
}

/*
* 생성된 노드의 수를 세는 함수
* count라는 노드를 세기 위한 변수를 선언
* cur이라는 헤드를 가르키는 포인터 ㅊㄱㅎ
* cur이 NULL 즉 끝이 아닐때까지
* 다음노드로 움직이면서
* count를 올리면서 셈
*/
// 노드 수 세기
int countNodes(D_Node* head) {
    int count = 0;
    D_Node* cur = head;

    while (cur != NULL) {
        count++;
        cur = cur->next;
    }

    return count;
}