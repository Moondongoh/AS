#ifndef FUN_H	
#define FUN_H

FILE* fp;
COORD pos = { 0,0 };	//커서 위치
long long max = 0;        // 입금
long long min = 0;        // 출금
long long balance = 0;    // 잔액
long long balance2 = 0;
int i = 0;
int Limit_min = INT_MIN;
int Limit_max = INT_MAX;
char str[100];
char str2[100];
char str3[100];
char str4[100];
char str5[100];
long long Sum_income = 0;
long long Sum_expense = 0;

void addCommas(char* str);
void cur(short x, short y);
void plus();
void minus();
void fixed();
void print();
void N_time();

#endif