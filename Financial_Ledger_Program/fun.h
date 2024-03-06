#ifndef FUN_H	
#define FUN_H

FILE* fp;
COORD pos = { 0,0 };        //커서 위치
long long max = 0;          // 입금
long long min = 0;          // 출금
long long balance = 0;      // 잔액
long long balance2 = 0;     // ,계산 후 잔액을 저장한 변수
int i = 0;
int Limit_min = INT_MIN;    // 최소 금액의 수치
int Limit_max = INT_MAX;    // 최고 금액의 수치
char str[100];              // itoa 함수 사용 후 저장한 변수
char str2[100];             // " "
char str3[100];             // " "
char str4[100];             // " "
char str5[100];             // " "
long long Sum_income = 0;   // 고정 수입 총 합
long long Sum_expense = 0;  // 고정 지출 총 합

void addCommas(char* str);  // 3자리씩 끊어서 ,를 추가하는 함수
void cur(short x, short y); // 초기화 시 이동한 위치 
void plus();                // 입금
void minus();               // 지출
void fixed();               // 고정 지출
void print();               // 잔액 출력
void N_time();              // 년 월 일 시 분 초 시간 나타내기

#endif
