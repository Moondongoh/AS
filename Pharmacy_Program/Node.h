//노드 정의//
typedef struct Node
{
	char name[50];
	int age;
	char phone[15];
	char address[100];
	int medicine_count;
	char medicine_names[10][50];
	struct Node* next;
}D_Node;