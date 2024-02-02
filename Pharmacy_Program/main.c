#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<windows.h>
#include "Fun_Def.h"
#include "Node.h"

//메인//

int main()
{
	D_Node* head = NULL;

	int C_num;

	while (1)
	{

		while (1)
		{
			printf("-----이용하실 서비스를 선택하세요!-----\n");
			printf("1.  새로운 환자 정보 입력(앞)\n");
			printf("2.  새로운 환자 정보 입력(뒤)\n");
			printf("3.  특정 환자 앞에 정보 입력\n");
			printf("4.  특정 환자 뒤에 정보 입력\n");
			printf("5.  맨 앞 환자 정보 삭제\n");
			printf("6.  맨 뒤 환자 정보 삭제\n");
			printf("7.  특정 환자 정보 삭제\n");
			printf("8.  모든 환자 정보 삭제\n");
			printf("9.  특정 환자 정보 수정\n");
			printf("10. 특정 환자 정보 읽기\n");
			printf("11. 모든 환자 정보 읽기\n");
			printf("12. 단일 선형 탐색으로 환자 정보 찾기\n");
			printf("13. 다중 선형 탐색으로 환자 정보 찾기\n");
			printf("14. 단일 이진 탐색으로 환자 정보 찾기\n");
			printf("15. 다중 이진 탐색으로 환자 정보 찾기\n");
			printf("16. 거품 정렬\n");
			printf("17. 삽입 정렬\n");
			printf("18. 선택 정렬\n");
			printf("20. 프로그램을 종료합니다.\n");
			printf("번호를 입력 해 주세요: ");

			if (scanf_s("%d", &C_num) != 1)
			{
				clear();
				printf("잘못된 값을 입력 했습니다\n");
				while (getchar() != '\n');
				continue;
			}
			printf("\n");
			break;
		}
		
		switch (C_num)
		{

		case 1:
			F_add_node(&head);
			printf("\n");
			break;

		case 2:
			B_add_node(&head);
			printf("\n");
			break;

		case 3:
			addBeforeNode(&head);
			printf("\n");
			break;

		case 4:
			addAfterNode(&head);
			printf("\n");
			break;

		case 5:
			// 앞 노드 데이터 삭제
			deleteFirstNode(&head);
			if (head != NULL)
			{
				clear();
			}

			else
			{
				clear();
				printf("현재 리스트가 비었습니다.\n");
				printf("\n");
			}
			break;

		case 6:
			// 뒤 노드 데이터 삭제
			deleteLastNode(&head);
			if (head != NULL)
			{
				clear();
			}

			else
			{
				clear();
				printf("현재 리스트가 비었습니다.\n");
				printf("\n");
			}
			break;

		case 7:
			deleteSearchNode(&head);
			clear();
			printf("\n");
			break;

		case 8:
			deleteALLNode(&head);
			printf("\n");
			break;

		case 9:
			ModifyNode(&head);
			printf("\n");
			break;

		case 10:
			PrintSearchNode(&head);
			printf("\n");
			break;

		case 11:
			printf("현재 환자 명단 상태: ");
			print_list(head);
			printf("\n");
			break;

		case 12:
			LinearSearch(&head);
			printf("\n");
			break;

		case 13:
			MultipleLinearSearch(&head);
			printf("\n");
			break;

		case 14:
			binarySearch(&head);
			printf("\n");
			break;

		case 15:
			multipleBinarySearch(&head);
			printf("\n");
			break;

		case 16:
			bubbleSort(&head);
			printf("버블 정렬이 완료되었습니다.\n");
			printf("현재 환자 명단 상태: ");
			print_list(head);
			printf("\n");
			break;

		case 17:
			insertionSort(&head);
			printf("\n");
			break;

		case 18:
			selectionSort(&head);
			printf("\n");
			break;

		case 20:
			break;

		default:
			clear();
			printf("잘못된 값을 입력 했습니다\n");
			while (getchar() != '\n');
			break;
		}
	}
}