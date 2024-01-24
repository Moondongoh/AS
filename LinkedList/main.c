#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<windows.h>
#include<limits.h>
#include"LinkedList.h"

void cur(short x, short y)		
{
    COORD pos = { x, y };
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
}

int main()
{
    D_Node* head = NULL;                                                    // 연결 리스트의 헤드 포인터 초기화
    D_Node* selectedNode = NULL;                                            //선택할 노드 포인터 초기화

    COORD pos = { 0,0 };//커서 위치

    int select;
    long long data; 
    int min = INT_MIN; // long long LLONG 수정
    int max = INT_MAX;

    printf("MIN : %d\n", min);
    printf("MAX : %d\n", max);

    while (1)
    {
        printf("-----이용하실 서비스를 선택하세요!-----\n");
        printf("1. 앞 노드 데이터 추가.\n");
        printf("2. 뒤 노드 데이터 추가.\n");
        printf("3. 기준 노드 앞 데이터 추가.\n");
        printf("4. 기준 노드 뒤 추가.\n");
        printf("5. 앞 노드 데이터 삭제.\n");
        printf("6. 뒤 노드 데이터 삭제.\n");
        printf("7. 임의 위치 데이터 삭제.\n");
        printf("8. 모든 노드 데이터 삭제.\n");
        printf("9. 임의 노드 데이터 수정.\n");
        printf("10. 임의 노드 데이터 값 읽기.\n");
        printf("11. 노드 순회하기.\n");
        printf("12. 단일 선형 탐색.\n");
        printf("13. 다중 선형 탐색.\n");
        printf("14. 단일 이진 탐색.\n");
        printf("15. 다중 이진 탐색.\n");
        printf("16. 거품 정렬.\n");
        printf("17. 삽입 정렬.\n");
        printf("18. 선택 정렬.\n");
        printf("19. 프로그램을 종료합니다.\n");
        printf("번호를 입력해 주세요 : ");
        scanf_s("%d", &select);
        printf("\n\n");

        if (select == 1)
        {
            // 앞 노드 데이터 추가
            printf("추가할 데이터를 입력하세요 :  ");
            // scanf ... !=1 은 정수가 아닌 값이 입력되었을때 검사한다.
            // 숫자가 아닌 다른 것을 입력 했을때 참이 됌.
            // 추가로 scanf는 정수가 입력되면 1 아니면 0을 반환함.
            // 즉 1이 아니다? 그러면 입력 버퍼 지워
            if (scanf("%lld", &data) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                while (getchar() != '\n');
                continue;
            }
            else if (data <= min)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("삐삑 오버플로우 발생했습니다 다시 하십쇼 휴먼\n");
                while (getchar() != '\n');
                continue;
            }
            else if (data >= max)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("삐삑 오버플로우 발생했습니다 다시 하십쇼 휴먼\n");
                while (getchar() != '\n');
                continue;
            }
            else
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                addFirstNode(&head, data);
                printf("현재 연결 리스트: ");
                printList(head);
                printf("\n");
            }
            //system("cls");
        }

        else if (select == 2)
        {
            // 뒤 노드 데이터 추가
            printf("추가할 데이터를 입력하세요 :  ");
            if (scanf("%lld", &data) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                while (getchar() != '\n');
                continue;
            }
            else if (data <= min)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("삐삑 오버플로우 발생했습니다 다시 하십쇼 휴먼\n");
                while (getchar() != '\n');
                continue;
            }
            else if (data >= max)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("삐삑 오버플로우 발생했습니다 다시 하십쇼 휴먼\n");
                while (getchar() != '\n');
                continue;
            }
            else
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                addLastNode(&head, data);
                printf("현재 연결 리스트: ");
                printList(head);
                printf("\n");
            }
            //system("cls");
        }

        else if (select == 3)
        {
            // 기준 노드 앞 데이터 추가
            printf("찾을 데이터를 입력하세요 :  ");
            long long searchData;
            scanf("%lld", &searchData);

            printf("추가할 데이터를 입력하세요 :  ");
            if (scanf("%lld", &data) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                while (getchar() != '\n');
                continue;
            }
            else if (data <= min)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("삐삑 오버플로우 발생했습니다 다시 하십쇼 휴먼\n");
                while (getchar() != '\n');
                continue;
            }
            else if (data >= max)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("삐삑 오버플로우 발생했습니다 다시 하십쇼 휴먼\n");
                while (getchar() != '\n');
                continue;
            }
            else
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                addBeforeNode(&head, searchData, data);
                printf("현재 연결 리스트: ");
                printList(head);
                printf("\n");
            }
            //system("cls");
        }

        else if (select == 4)
        {
            // 기준 노드 뒤 데이터 추가
            printf("찾을 데이터를 입력하세요 :  ");
            long long searchDataAfter;
            scanf("%lld", &searchDataAfter);

            printf("추가할 데이터를 입력하세요 :  ");
            if (scanf("%lld", &data) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                while (getchar() != '\n');
                continue;
            }
            else if (data <= min)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("삐삑 오버플로우 발생했습니다 다시 하십쇼 휴먼\n");
                while (getchar() != '\n');
                continue;
            }
            else if (data >= max)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("삐삑 오버플로우 발생했습니다 다시 하십쇼 휴먼\n");
                while (getchar() != '\n');
                continue;
            }
            else
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                addAfterNode(head, searchDataAfter, data);
                printf("현재 연결 리스트: ");
                printList(head);
                printf("\n");
            }
            //system("cls");
        }
        else if (select == 5)
        {
            // 앞 노드 데이터 삭제
            deleteFirstNode(&head);
            if (head != NULL)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("현재 연결 리스트: ");
                printList(head);
                printf("\n");
            }
            //system("cls");
            else
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("현재 리스트가 비었습니다.\n");
            }
        }
        else if (select == 6)
        {
            // 뒤 노드 데이터 삭제
            deleteLastNode(&head);
            if (head != NULL)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("현재 연결 리스트: ");
                printList(head);
                printf("\n");
            }
            //system("cls");
            else
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("현재 리스트가 비었습니다.\n");
            }
        }
        else if (select == 7)
        {
            // 임의 위치 데이터 삭제
            int position = countNodes(head);
            printf("삭제할 위치를 입력하세요: ");
            if (scanf("%d", &position) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                while (getchar() != '\n');
                continue;
            }

            deleteSearchNode(&head, position);
            if (head != NULL)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("현재 연결 리스트: ");
                printList(head);
                printf("\n");
            }
            //system("cls");
            else
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("현재 리스트가 비었습니다.\n");
            }
        }
        else if (select == 8)
        {
            // 모든 노드 데이터 삭제
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            deleteALLNode(&head);
            printf("모두 삭제 되었습니다.\n");
            printf("현재 연결 리스트 : ");
            printList(head);
            //system("cls");
        }
        else if (select == 9)
        {
            // 임의 노드 데이터 수정
            int position3;
            printf("수정할 노드의 위치를 입력하세요: ");
            if (scanf("%d", &position3) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                while (getchar() != '\n');
                continue;
            }

            ModifyNode(head);
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            printf("현재 연결 리스트: ");
            printList(head);
            continue;
            //system("cls");
        }
        else if (select == 10)
        {
            // 임의 노드 데이터 값 읽기
            int position2;
            printf("출력할 노드의 위치를 입력하세요: ");
            if (scanf("%d", &position2) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                while (getchar() != '\n');
                continue;
            }
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            PrintSearchNode(head, position2);
            printf("현재 연결 리스트: ");
            printList(head);

            continue;
            //system("cls");
        }
        else if (select == 11)
        {
            // 노드 순회하기
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            printf("현재 연결 리스트: ");
            printList(head);
            printf("\n");
            continue;
            //system("cls");
        }
        else if (select == 12)
        {
            int position4 = countNodes(head);
            // 단일 선형 탐색
            int searchData;
            printf("찾을 값을 입력하세요: ");
            if (scanf("%d", &searchData) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                printf("현재 연결 리스트: ");
                printList(head);
                while (getchar() != '\n');
                continue;
            }
            system("cls");

            LinearSearch(head, position4,searchData);
            printf("현재 연결 리스트: ");
            printList(head);

        }
        else if (select == 13)
        {
            // 다중 선형 탐색
            int searchData;
            printf("찾을 값을 입력하세요: ");
            if (scanf("%d", &searchData) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                printf("현재 연결 리스트: ");
                printList(head);
                while (getchar() != '\n');
                continue;
            }
            system("cls");

            MultipleLinearSearch(head, searchData);
            printf("현재 연결 리스트: ");
            printList(head);
            continue;
            //system("cls");
        }

        else if (select == 14)
        {
            // 단일 이진 탐색
            bubbleSort(&head);  //정렬 시켜야함 이진 탐색ㅇㅇ
            printf("버블 정렬이 완료되었습니다.\n");

            int searchData;
            printf("찾을 값을 입력하세요: ");
            if (scanf("%d", &searchData) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                printf("현재 연결 리스트: ");
                printList(head);
                while (getchar() != '\n');
                continue;
            }
            system("cls");
            binarySearch(head, searchData);

            printf("현재 연결 리스트: ");
            printList(head);
            continue;
            //system("cls");
        }

        else if (select == 15)
        {
            // 다중 이진 탐색
            bubbleSort(&head);  //정렬 시켜야함 이진 탐색ㅇㅇ
            printf("버블 정렬이 완료되었습니다.\n");

            int searchData;
            printf("찾을 값을 입력하세요: ");
            if (scanf("%d", &searchData) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다\n");
                printf("현재 연결 리스트: ");
                printList(head);
                while (getchar() != '\n');
                continue;
            }
            system("cls");

            multipleBinarySearch(head, searchData);

            printf("현재 연결 리스트: ");
            printList(head);
            continue;
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);

        }
        else if (select == 16)
        {
            // 버블정렬
            bubbleSort(&head);
            printf("버블 정렬이 완료되었습니다.\n");
            system("cls");
            printf("현재 연결 리스트: ");
            printList(head);
            continue;
            //system("cls");
        }
        else if (select == 17)
        {
            // 삽입정렬
            insertionSort(&head);
            printf("삽입 정렬이 완료되었습니다.\n");
            system("cls");
            printf("현재 연결 리스트: ");
            printList(head);
            continue;
            //system("cls");
        }
        else if (select == 18)
        {
            // 선택정렬
            selectionSort(&head);
            printf("선택 정렬이 완료되었습니다.\n");
            system("cls");
            printf("현재 연결 리스트: ");
            printList(head);
            continue;
            //system("cls");
        }
        else if (select == 19)
        {
            // 종료
            break;
        }
        else
        {
            system("cls");
            printf("잘못된 선택입니다. 다시 선택하세요.\n");
            //내가 만약에 숫자가 아닌 문자를 집어 넣음 그러면 while문을
            //안쓰면 select는 scanf에서 숫자를 인식을 못해서 값이 없게됌.
            //측 select 변수는 초기화 상태로 남아 있는 거임.
            //그래서 입력 버퍼를 지워주고 사용자에게 다시 입력을 받도록 컨티뉴함.
            while (getchar() != '\n');
            continue;
        }
    }

    // 메모리 해제
    // head값이 NULL이 아닐때 동안 계속 while문 실행
    // cur을 head로 초기화해 현재 노드를 가르키도록
    // head를 다음 노드로 이동 후
    // 현재 노드를 메모리 해제함.
    while (head != NULL)
    {
        struct Node* cur = head;
        head = head->next;
        free(cur);
    }

    return 0;
}
