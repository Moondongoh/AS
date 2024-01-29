#ifndef _LinkedList_H_
#define _LinkedList_H_
#include <stdio.h>
#include"Node.h"

//함수 선언//
void F_add_node(D_Node** head);						//1번  C
void B_add_node(D_Node** head);						//2번  C
void addBeforeNode(D_Node** head);                  //3번  C
void addAfterNode(D_Node** head);					//4번  C
void deleteFirstNode(D_Node** head);				//5번  C
void deleteLastNode(D_Node** head);					//6번  C
void deleteSearchNode(D_Node** head);				//7번  C
void deleteALLNode(D_Node** head);					//8번  C
void ModifyNode(D_Node** head);						//9번  C
void PrintSearchNode(D_Node** head);				//10번 C
void print_list(D_Node* head);						//11번 C
void LinearSearch(D_Node** head);					//12번 C
void MultipleLinearSearch(D_Node** head);			//13번 C
void binarySearch(D_Node** head);					//14번 C
void multipleBinarySearch(D_Node** head);			//15번 C
void bubbleSort(D_Node** head);						//16번 C
void insertionSort(D_Node** head);					//17번 C
void selectionSort(D_Node** head);					//18번 C
void clear();

// *예외처리* 같은 약의 이름을 찾게 되면 다시 입력 받게함.
int check_repeat(const char name[][50], int count, const char* new_name);

int check_repeat(const char name[][50], int count, const char* new_name)
{
	for (int i = 0; i < count; ++i)
	{
		if (strcmp(name[i], new_name) == 0)
		{
			return 1;
		}
	}
	return 0;
}

//함수//

void clear()
{
	COORD pos = { 0, 0 };
	system("cls");
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
}

D_Node* create_node()
	{
		COORD pos = { 0,0 };

		D_Node* new_node = (D_Node*)malloc(sizeof(D_Node));

		printf("이름을 입력하세요: ");
		scanf_s(" %49[^\n]s", &new_node->name, sizeof(new_node->name));

		printf("나이를 입력하세요: ");
		while (1)
		{
			if (scanf_s("%d", &new_node->age) != 1)
			{
				system("cls");
				SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
				printf("잘못된 값을 입력 했습니다\n");
				printf("다시 나이를 입력 해주세요. : ");
				while (getchar() != '\n');
				continue;
			}
			break;
		}

		printf("전화번호를 입력하세요: ");
		scanf_s(" %14[^\n]s", &new_node->phone, sizeof(new_node->phone));

		printf("주소를 입력하세요: ");
		scanf_s(" %99[^\n]s", &new_node->address, sizeof(new_node->address));

		printf("약의 갯수를 입력하세요: ");
		while (1)
		{
			if (scanf_s("%d", &new_node->medicine_count) != 1)
			{
				system("cls");
				SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
				printf("잘못된 값을 입력 했습니다\n");
				printf("다시 약의 갯수를 입력 해주세요. : ");
				while (getchar() != '\n');
				continue;
			}
			break;
		}

		printf("약의 이름을 입력하세요\n");
		for (int i = 0; i < new_node->medicine_count; ++i)
		{
			do {
				printf("약 %d: ", i + 1);
				scanf_s(" %49[^\n]s", new_node->medicine_names[i], sizeof(new_node->medicine_names[i]));

				if (check_repeat(new_node->medicine_names, i, new_node->medicine_names[i])) 
				{
					system("cls");
					SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
					printf("약의 이름이 중복되었습니다. 다시 입력해주세요.\n");
				}
			} while (check_repeat(new_node->medicine_names, i, new_node->medicine_names[i]));
		}

		new_node->next = NULL;
		return new_node;
	}

// 1.  새로운 환자 정보 입력(앞)
void F_add_node(D_Node** head)
{
	clear();
	D_Node* new_node = create_node();
	new_node->next = *head;
	*head = new_node;
	clear();
}

// 2.  새로운 환자 정보 입력(뒤)
void B_add_node(D_Node** head)
{
	clear();
	D_Node* new_node = create_node();

	if (*head == NULL)
	{
		*head = new_node;
	}

	else 
	{
		D_Node* cur = *head;
		while (cur->next != NULL)
		{
			cur = cur->next;
		}
		cur->next = new_node;
	}
	clear();
}

// 3.  특정 환자 앞에 정보 입력
void addBeforeNode(D_Node** head)
{
	clear();
	D_Node* cur = *head;
	D_Node* prev = NULL;

	int position3;
	int count = 0;

	printf("몇번 환자 앞에 데이터를 추가하시겠습니까? : ");
	if (scanf("%d", &position3) != 1)
	{
		printf("잘못된 값을 입력 했습니다\n");
		while (getchar() != '\n');
	}

	//노드의 데이터가 searchData와 일치하는 노드를 찾음
	while (cur != NULL && count < position3)
	{
		prev = cur;
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
		// 새로운 노드 동적 할당
		D_Node* new_node = create_node();

		// 이전 노드가 없는 경우 (즉, 찾은 노드가 헤드인 경우)
		if (prev == NULL)
		{
			new_node->next = *head;
			*head = new_node; // 헤드를 새로운 노드로 업데이트
		}

		// 이전 노드가 있는 경우
		else
		{
			new_node->next = prev->next;
			prev->next = new_node;
		}
	}
	clear();
}

// 4.  특정 환자 뒤에 정보 입력
void addAfterNode(D_Node** head)
{
	clear();

	D_Node* cur = *head;
	D_Node* prev = NULL;

	int position4;
	int count = 0;

	printf("몇번 환자 뒤에 데이터를 추가하시겠습니까? : ");
	if (scanf("%d", &position4) != 1)
	{
		printf("잘못된 값을 입력 했습니다\n");
		while (getchar() != '\n');
	}

	//노드의 데이터가 searchData와 일치하는 노드를 찾음
	while (cur != NULL && count < position4)
	{
		prev = cur;
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
		// 새로운 노드 동적 할당
		D_Node* new_node = create_node();
		new_node->next = cur->next;
		cur->next = new_node;
	}
	clear();
}

// 5.  맨 앞 환자 정보 삭제
void deleteFirstNode(D_Node** head)
{
	if (*head == NULL) 
	{
		printf("삭제할 노드가 없습니다.\n");
		return;
	}

	D_Node* new_head = (*head)->next;
	free(*head);
	*head = new_head;
}

// 6.  맨 뒤 환자 정보 삭제
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

// 7.  특정 환자 정보 삭제
void deleteSearchNode(D_Node** head)
{
	clear();
	D_Node* cur = *head;
	D_Node* prev = NULL;

	int position3;
	int count = 0;

	printf("몇 번 환자의 데이터를 삭제하시겠습니까? (0부터 시작): ");
	if (scanf("%d", &position3) != 1)
	{
		printf("잘못된 값을 입력 했습니다\n");
		while (getchar() != '\n');
	}

	if (*head == NULL)
	{
		printf("삭제할 노드가 없습니다.\n");
		return;
	}

	if (position3 == 0)
	{
		// 헤드 노드를 삭제하는 경우
		D_Node* temp = *head;
		*head = (*head)->next;
		free(temp);
		return;
	}

	// 노드의 위치를 찾음
	while (cur != NULL && count < position3)
	{
		prev = cur;
		cur = cur->next;
		count++;
	}

	if (cur == NULL)
	{
		// searchData와 일치하는 노드를 찾지 못한 경우
		printf("일치하는 노드를 찾을 수 없습니다.\n");
		return;
	}

	// 찾은 노드를 삭제
	D_Node* temp = cur;
	prev->next = cur->next;
	free(temp);
	return;
}

// 8.  모든 환자 정보 삭제
void deleteALLNode(D_Node** head)
{
	clear();
	D_Node* prev = *head;

	while (*head)
	{
		*head = (*head)->next;

		//printf("삭제할 데이터 값 : %d\n", prev->data);
		free(prev);
		prev = *head;
	}
	clear();
}

// 9.  특정 환자 정보 수정
void ModifyNode(D_Node** head)
{
	clear();
	COORD pos = { 0,0 };

	D_Node* cur = *head;
	D_Node* prev = NULL;

	int position3;
	int count = 0;

	printf("몇번 환자의 데이터를 수정하시겠습니까? : ");
	if (scanf("%d", &position3) != 1)
	{
		printf("잘못된 값을 입력 했습니다\n");
		while (getchar() != '\n');
	}

	//노드의 데이터가 searchData와 일치하는 노드를 찾음
	while (cur != NULL && count < position3)
	{
		prev = cur;
		cur = cur->next;
		count++;
	}

	// searchData와 일치하는 노드를 찾지 못한 경우
	if (cur == NULL)
	{
		printf("일치하는 노드를 찾을 수 없습니다.\n");
		return;
	}

	// 헤드 노드를 삭제하는 경우
	if (position3 == 0)
	{
		D_Node* temp = *head;
		*head = (*head)->next;
		free(temp);
	}

	// 찾은 노드를 삭제
	else
	{
		D_Node* temp = cur;
		prev->next = cur->next;
		free(temp);
	}

	// 새로운 노드 동적 할당
	D_Node* new_node = create_node();

	// 이전 노드가 없는 경우 (즉, 찾은 노드가 헤드인 경우)
	if (prev == NULL)
	{
		new_node->next = *head;
		*head = new_node; // 헤드를 새로운 노드로 업데이트
	}

	// 이전 노드가 있는 경우
	else
	{
		new_node->next = prev->next;
		prev->next = new_node;
	}
	clear();
}

// 10. 특정 환자 정보 읽기
void PrintSearchNode(D_Node** head)
{
	clear();
	D_Node* cur = *head;
	D_Node* prev = NULL;

	int position3;
	int count = 0;

	printf("몇번 환자의 데이터를 확인하시겠습니까? : ");
	if (scanf("%d", &position3) != 1)
	{
		printf("잘못된 값을 입력 했습니다\n");
		while (getchar() != '\n');
	}

	while (cur != NULL && count < position3)
	{
		prev = cur;
		cur = cur->next;
		count++;
	}

	if (cur != NULL)
	{
		printf("해당 번호 환자 정보: \n이름: %s\n나이: %d\n전화번호: %s\n주소: %s\n약의 갯수: %d\n약의 이름: %s",
			cur->name, cur->age, cur->phone, cur->address, cur->medicine_count, cur->medicine_names);
	}

	// searchData와 일치하는 노드를 찾지 못한 경우
	if (cur == NULL)
	{
		printf("일치하는 노드를 찾을 수 없습니다.\n");
		return;
	}
}

// 11. 모든 환자 정보 읽기
void print_list(D_Node* head)
{
	clear();
	D_Node* cur = head;

	if (head == NULL)
	{
		printf("환자 정보가 없습니다.");
	}

	while (cur != NULL) 
	{
		printf("\n이름: %s\n나이: %d\n전화번호: %s\n주소: %s\n약의 갯수: %d\n약의 이름: ",
			cur->name, cur->age, cur->phone, cur->address, cur->medicine_count);

		for (int i = 0; i < cur->medicine_count; ++i)
		{
			printf("%s ", cur->medicine_names[i]);
		}

		printf("\n\n");
		cur = cur->next;
	}
}

// 12. 단일 선형 탐색으로 환자 정보 찾기
void LinearSearch(D_Node** head)
{
	clear();
	D_Node* cur = *head;

	int count = 0;
	char searchData[50];

	printf("찾으실 환자 이름을 입력해주세요: ");
	scanf_s("%49s", searchData, sizeof(searchData));
	while (cur != NULL && strcmp(cur->name, searchData) != 0)
	{
		cur = cur->next;
		count++;
	}

	if (cur == NULL)
	{
		printf("일치하는 노드를 찾을 수 없습니다.\n");
	}

	else
	{
		printf("%s 환자의 위치는: %d\n", searchData, count+1);
	}
}

// 13. 다중 선형 탐색으로 환자 정보 찾기
void MultipleLinearSearch(D_Node** head)
{
	clear();
	D_Node* cur = *head;

	int count = 0;
	char searchData[50];

	printf("찾으실 환자 이름을 입력해주세요: ");
	scanf_s("%49s", searchData, sizeof(searchData));

	while (cur != NULL)
	{
		if (strcmp(cur->name, searchData) == 0)
		{
			printf("%s 환자의 위치는: %d\n", searchData, count+1);
		}
		cur = cur->next;
		count++;
	}

	if (count == 0)
	{
		printf("일치하는 노드를 찾을 수 없습니다.\n");
	}
}

// 14. 단일 이진 탐색으로 환자 정보 찾기
void binarySearch(D_Node** head)
{
	clear();
	// 탐색을 위해 노드 정렬 시키기
	bubbleSort(head);
	printf("정렬이 완료 되었습니다.\n");

	int count = 0;  // 노드 수 셀 변수
	int position = -1;
	char searchData[50];

	printf("찾으실 환자 이름을 입력해주세요: ");
	scanf_s("%49s", searchData, sizeof(searchData));

	// 리스트의 노드 수 계산
	D_Node* temp = *head;
	while (temp != NULL) {
		count++;
		temp = temp->next;
	}

	// 노드의 데이터가 searchData와 일치하는 노드를 찾음
	int low = 0;
	int high = count - 1;

	while (low <= high) 
	{
		int mid = (low + high) / 2;

		D_Node* cur = *head;

		for (int i = 0; i < mid; i++) 
		{
			cur = cur->next;
		}

		if (strcmp(cur->name, searchData) == 0) 
		{
			position = mid;
			break;
		}

		else if (strcmp(cur->name, searchData) < 0) 
		{
			low = mid + 1;
		}

		else 
		{
			high = mid - 1;
		}
	}

	if (position != -1) 
	{
		printf("%s 환자의 순서는: %d번 입니다.\n", searchData, position + 1);
	}

	// searchData와 일치하는 노드를 찾지 못한 경우
	else 
	{
		printf("일치하는 노드를 찾을 수 없습니다.\n");
	}
}

// 15. 다중 이진 탐색으로 환자 정보 찾기
void multipleBinarySearch(D_Node** head)
{
	clear();
	// 탐색을 위해 노드 정렬 시키기
	bubbleSort(head);
	printf("정렬이 완료 되었습니다.\n");

	int count = 0;  // 노드 수 셀 변수
	int position = -1;
	char searchData[50];

	printf("찾으실 환자 이름을 입력해주세요: ");
	scanf_s("%49s", searchData, sizeof(searchData));

	// 리스트의 노드 수 계산
	D_Node* temp = *head;
	while (temp != NULL)
	{
		count++;
		temp = temp->next;
	}

	// 노드의 데이터가 searchData와 일치하는 노드를 찾음
	int low = 0;
	int high = count - 1;

	while (low <= high)
	{
		int mid = (low + high) / 2;

		D_Node* cur = *head;
		for (int i = 0; i < mid; i++)
		{
			cur = cur->next;
		}

		if (strcmp(cur->name, searchData) == 0)
		{
			position = mid;
			printf("%s 값의 노드 위치는: %d\n", searchData, position+1);

			// 왼쪽으로 이동하며 출력
			int left = position - 1;

			D_Node* leftNode = *head; // 수정: head로 초기화
			while (left >= 0 && strcmp(leftNode->name, searchData) == 0)
			{
				printf("%s 값의 노드 위치는: %d\n", searchData, left+1);
				left--;
				leftNode = leftNode->next;
			}

			// 오른쪽으로 이동하며 출력
			int right = position + 1;

			D_Node* rightNode = cur->next; // 현재 위치에서의 다음 노드부터 시작
			while (right < count && strcmp(rightNode->name, searchData) == 0)
			{
				printf("%s 값의 노드 위치는: %d\n", searchData, right+1);
				right++;
				rightNode = rightNode->next;
			}

			return;  // 찾은 후에는 더 이상 진행하지 않음
		}

		else if (strcmp(cur->name, searchData) < 0) // 수정: 문자열 비교로 변경
		{
			low = mid + 1;
		}

		else
		{
			high = mid - 1;
		}
	}
	// searchData와 일치하는 노드를 찾지 못한 경우
	printf("일치하는 노드를 찾을 수 없습니다.\n");
}

// 16. 거품 정렬
void bubbleSort(D_Node** head)
{
	clear();
	D_Node* cur;
	D_Node* last = NULL;

	int swapped;

	// 빈 리스트 또는 하나의 노드만 있는 경우 정렬할 필요 없음
	if (*head == NULL || (*head)->next == NULL)
	{
		return;
	}

	do
	{
		swapped = 0;
		cur = *head;
		D_Node* prev = NULL;

		while (cur->next != last)
		{
			// 현재 노드와 다음 노드의 이름을 비교하여 교환
			if (strcmp(cur->name, cur->next->name) > 0) 
			{
				// 이름이 더 크다면 노드의 주소를 교환
				D_Node* temp = cur->next;
				cur->next = temp->next;
				temp->next = cur;

				// 헤드를 바꾸어야 하는 경우
				if (prev == NULL)
				{
					*head = temp;
				}

				else
				{
					prev->next = temp;
				}

				cur = temp;
				swapped = 1;
			}

			prev = cur;
			cur = cur->next;
		}

		last = cur;

	} while (swapped);
	clear();
}

// 17. 삽입 정렬
void insertionSort(D_Node** head)
{
	clear();
	D_Node* sorted = NULL;

	int standard;

	printf("기준을 선택하기: \n");
	printf("1: 이름\n");
	printf("2: 나이\n");
	printf("3: 주소\n");
	printf("1,2,3번 중 번호 선택하기:");
	scanf_s("%d", &standard);

	while (*head != NULL)
	{
		D_Node* cur = *head;
		*head = (*head)->next;

		if (standard == 1)
		{

			if (sorted == NULL || strcmp(cur->name, sorted->name) < 0)
			{
				cur->next = sorted;
				sorted = cur;
			}

			else
			{
				D_Node* temp = sorted;
				while (temp->next != NULL && strcmp(cur->name, temp->next->name) >= 0)
				{
					temp = temp->next;
				}

				cur->next = temp->next;
				temp->next = cur;
			}
		}

		else if (standard == 2)
		{
			if (sorted == NULL || cur->age < sorted->age)
			{
				cur->next = sorted;
				sorted = cur;
			}

			else 
			{
				D_Node* temp = sorted;
				while (temp->next != NULL && cur->age >= temp->next->age)
				{
					temp = temp->next;
				}

				cur->next = temp->next;
				temp->next = cur;
			}
		}

		else
		{
			if (sorted == NULL || strcmp(cur->address, sorted->address) < 0)
			{
				cur->next = sorted;
				sorted = cur;
			}

			else
			{
				D_Node* temp = sorted;
				while (temp->next != NULL && strcmp(cur->address, temp->next->address) >= 0)
				{
					temp = temp->next;
				}

				cur->next = temp->next;
				temp->next = cur;
			}
		}
	}

	*head = sorted;
	clear();
}

// 18. 선택 정렬
void selectionSort(D_Node** head)
{
	clear();
	D_Node* cur = *head;
	D_Node* temp = NULL;
	D_Node* sorted = NULL;

	int standard;

	printf("기준을 선택하기: \n");
	printf("1: 이름\n");
	printf("2: 나이\n");
	printf("3: 주소\n");
	printf("1,2,3번 중 번호 선택하기:");
	scanf_s("%d", &standard);

	while (cur != NULL)
	{
		D_Node* minNode = cur;
		D_Node* searchNode = cur->next;

		if (standard == 1)
		{
			// 최솟값 찾기
			while (searchNode != NULL)
			{
				if (strcmp(searchNode->name, minNode->name) < 0)
				{
					minNode = searchNode;  // 최솟값을 찾으면 minNode를 업데이트
				}
				searchNode = searchNode->next;
			}

			// 최솟값과 현재 노드의 데이터 교환 (만약 minNode가 업데이트된 경우에만 교환)
			if (minNode != cur)
			{
				// 문자열을 교환하는 대신, 데이터를 복사합니다.
				char tempData[100];  // 적절한 최대 길이를 가정합니다.
				strcpy(tempData, cur->name);
				strcpy(cur->name, minNode->name);
				strcpy(minNode->name, tempData);
			}

			cur = cur->next;
		}

		if (standard == 2)
		{
			// 최솟값 찾기
			while (searchNode != NULL)
			{
				if (searchNode->age < minNode->age)
				{
					minNode = searchNode;
				}
				searchNode = searchNode->next;
			}

			// 최솟값과 현재 노드의 데이터 교환
			if (minNode != cur)
			{
				int tempData = cur->age;
				cur->age = minNode->age;
				minNode->age = tempData;
			}

			cur = cur->next;
		}

		if (standard == 3)
		{

			// 최솟값 찾기
			while (searchNode != NULL)
			{
				if (strcmp(searchNode->address, minNode->address) < 0)
				{
					minNode = searchNode;  // 최솟값을 찾으면 minNode를 업데이트
				}
				searchNode = searchNode->next;
			}

			// 최솟값과 현재 노드의 데이터 교환 (만약 minNode가 업데이트된 경우에만 교환)
			if (minNode != cur)
			{
				// 문자열을 교환하는 대신, 데이터를 복사합니다.
				char tempData[100];  // 적절한 최대 길이를 가정합니다.
				strcpy(tempData, cur->address);
				strcpy(cur->address, minNode->address);
				strcpy(minNode->address, tempData);
			}

			cur = cur->next;
		}


	}
	clear();
}

// 19. 메모리 반납 
void free_list(D_Node* head)
{
	D_Node* cur = head;
	D_Node* next_node;
	while (cur != NULL)
	{
		next_node = cur->next;
		free(cur);
		cur = next_node;
	}
}
#endif951