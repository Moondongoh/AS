#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <limits.h>
#include "fun.h"
#pragma warning(disable:4996)

void cur(short x, short y) //화면 초기화 시 사용 될 좌표 값
{
    COORD pos = { x, y };
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
}

void addCommas(char* str) // 0을 3개씩 끊어서 ,를 추가하는 함수
{
    int len = strlen(str);
    int numCommas = (len - 1) / 3;
    int newLen = len + numCommas;

    char* result = (char*)malloc(newLen + 1); // 널 종료 문자를 위한 공간 확보 (+1)
    int i, j;

    for (i = 0, j = 0; i < len; ++i)
    {
        if (i > 0 && (len - i) % 3 == 0)
        {
            result[j++] = ',';
        }
        result[j++] = str[i];
    }

    result[j] = '\0'; // 널 종료 문자

    strcpy(str, result);

    free(result);
}

void plus()
{
    i = 0;
    char income[15];
    time_t timer;
    struct tm* t;
    timer = time(NULL);
    t = localtime(&timer);
    printf("입금 내역을 적어주세요: ");
    scanf_s("%s", &income, 15);
    while (1)
    {
        printf("입금 금액: ");
        if (scanf("%lld", &max) != 1)
        {
            printf("잘못된 값을 입력 했습니다\n");
            while (getchar() != '\n');
            continue;
        }
        else if (max < Limit_min)
        {
            printf("언더 플로우가 발생하였습니다. 다시 입력해주세요.\n");
            while (getchar() != '\n');
            continue;
        }
        else if (max > Limit_max)
        {
            printf("오버 플로우가 발생하였습니다. 다시 입력해주세요.\n");
            while (getchar() != '\n');
            continue;
        }

        itoa(max, str, 10);
        addCommas(str); // 쉼표를 추가하여 형식화
        balance2 += max;

        itoa(balance2, str2, 10);
        addCommas(str2); // 쉼표를 추가하여 형식화

        printf("거래 중입니다.");
        for (i; i < 3;i++)
        {
            printf("->");
            Sleep(300);
        }
        printf("거래 완료!");
        fprintf(fp, "\n입금 내역: %s\n입금 금액: %s원\n계좌 잔액: %s원\n거래시간: %d년 %d월 %d일 %d시 %d분 %d초\n", income, str, str2, t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
            t->tm_hour, t->tm_min, t->tm_sec);
        printf("\n잔액: %s원\n", str2);
        printf("처리되었습니다.\n");
        N_time();
        break;
    }
}

void minus()
{
    i = 0;
    char expense[15];
    time_t timer;
    struct tm* t;
    timer = time(NULL);
    t = localtime(&timer);
    printf("지출 내역을 적어주세요: ");
    scanf_s("%s", &expense, 15);
    while (1)
    {
        printf("지출 금액: ");
        if (scanf("%lld", &min) != 1)
        {
            printf("잘못된 값을 입력 했습니다\n");
            while (getchar() != '\n');
            continue;
        }
        else if (min < Limit_min)
        {
            printf("언더 플로우가 발생하였습니다. 다시 입력해주세요.\n");
            while (getchar() != '\n');
            continue;
        }
        else if (min > Limit_max)
        {
            printf("오버 플로우가 발생하였습니다. 다시 입력해주세요.\n");
            while (getchar() != '\n');
            continue;
        }

        itoa(min, str, 10);
        addCommas(str); // 쉼표를 추가하여 형식화
        balance2 -= min;

        itoa(balance2, str2, 10);
        addCommas(str2); // 쉼표를 추가하여 형식화

        printf("거래 중입니다.");
        for (i; i < 3;i++)
        {
            printf("->");
            Sleep(300);
        }
        printf("거래 완료!");
        fprintf(fp, "\n지출 내역: %s\n지출 금액: %s원\n계좌 잔액: %s원\n거래시간: %d년 %d월 %d일 %d시 %d분 %d초\n", expense, str, str2, t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
            t->tm_hour, t->tm_min, t->tm_sec);
        printf("\n잔액: %s\n", str2);
        printf("처리되었습니다.\n");
        N_time();
        break;
    }

}

void fixed()
{
    long long F_income = 0;
    int F_income_C = 0;
    char F_income_N[100];
    long long F_expense = 0;
    int F_expense_C = 0;
    char F_expense_N[100];
    while (1)
    {
        int F_choice;
        while (1)
        {
            printf("\n추가할 고정 수입/지출을 선택해 주세요.\n1.고정 수입\n2.고정 지출\n3.건너뛰기\n번호 선택 :");
            if (scanf("%d", &F_choice) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다.\n");
                printf("추가하실 내역를 선택해주세요.\n");
                while (getchar() != '\n');
                continue;
            }
            break;
        }

        switch (F_choice)
        {
        case 1:
            fprintf(fp, "===================고정 수입===================\n");
            while (1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("입력하실 고정 수입 내역을 입력해주세요:");
                scanf("%s", &F_income_N);
                while (1)
                {
                    printf("고정 수입의 금액을 입력해주세요: ");
                    if (scanf("%lld", &F_income) != 1)
                    {
                        system("cls");
                        SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                        printf("잘못된 값을 입력 했습니다.\n");
                        printf("고정 수입을 숫자로 입력해주세요.\n");
                        while (getchar() != '\n');
                        continue;
                    }
                    else if (F_income < Limit_min)
                    {
                        printf("언더 플로우가 발생하였습니다. 다시 입력해주세요.\n");
                        while (getchar() != '\n');
                        continue;
                    }
                    else if (F_income > Limit_max)
                    {
                        printf("오버 플로우가 발생하였습니다. 다시 입력해주세요.\n");
                        while (getchar() != '\n');
                        continue;
                    }
                    break;
                }
                itoa(F_income, str3, 10);
                addCommas(str3); // 쉼표를 추가하여 형식화
                printf("입력하신 내역과 금액은 %s,%s원 입니다.\n", F_income_N, str3);
                Sum_income += F_income;
                printf("더 추가 하시겠습니다?\n1.네\n2.아니요\n번호 선택: ");
                scanf("%d", &F_income_C);
                switch (F_income_C)
                {
                case 1:
                    system("cls");
                    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                    fprintf(fp, "내역: %s\n금액: %s원\n", F_income_N, str3);
                    continue;
                case 2:
                    system("cls");
                    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                    itoa(Sum_income, str4, 10);
                    addCommas(str4); // 쉼표를 추가하여 형식화
                    fprintf(fp, "내역: %s\n금액: %s원\n총 고정 수입 금액: %s원\n", F_income_N, str3, str4);
                    break;
                dafault:
                    system("cls");
                    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                    printf("1,2,3중에서 선택해주세요.");
                    continue;
                }
                break;
            }
            break;
        case 2:
            fprintf(fp, "===================고정 지출===================\n");
            while (1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("입력하실 고정 지출 내역을 입력해주세요:");
                scanf("%s", &F_expense_N);
                while (1)
                {
                    printf("고정 지출의 금액을 입력해주세요: ");
                    if (scanf("%lld", &F_expense) != 1)
                    {
                        system("cls");
                        SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                        printf("잘못된 값을 입력 했습니다.\n");
                        printf("고정 수입을 숫자로 입력해주세요.\n");
                        while (getchar() != '\n');
                        continue;
                    }
                    else if (F_expense < Limit_min)
                    {
                        printf("언더 플로우가 발생하였습니다. 다시 입력해주세요.\n");
                        while (getchar() != '\n');
                        continue;
                    }
                    else if (F_expense > Limit_max)
                    {
                        printf("오버 플로우가 발생하였습니다. 다시 입력해주세요.\n");
                        while (getchar() != '\n');
                        continue;
                    }
                    break;
                }
                itoa(F_expense, str3, 10);
                addCommas(str3); // 쉼표를 추가하여 형식화
                printf("입력하신 내역과 금액은 %s,%s원 입니다.\n", F_expense_N, str3);
                Sum_expense += F_expense;
                printf("더 추가 하시겠습니다?\n1.네\n2.아니요\n번호 선택: ");
                scanf("%d", &F_expense_C);
                switch (F_expense_C)
                {
                case 1:
                    system("cls");
                    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                    fprintf(fp, "내역: %s\n금액: %s원\n", F_expense_N, str3);
                    continue;
                case 2:
                    system("cls");
                    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                    itoa(Sum_expense, str4, 10);
                    addCommas(str4); // 쉼표를 추가하여 형식화
                    fprintf(fp, "내역: %s\n금액: %d원\n총 고정 지출 금액: %s원\n", F_expense_N, F_expense, str4);
                    break;
                default:
                    system("cls");
                    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                    printf("1,2,3 중에서 선택해주세요.\n");
                    continue;  // 다시 입력 요구
                }
                break;
            }
            break;
        case 3:
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            break;
        default:
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            printf("1,2,3중에서 선택해주세요.");
            break;
        }
        if (F_choice == 3)
            break;
    }
}

void print()
{
    i = 0;
    time_t timer;
    struct tm* t;
    timer = time(NULL);
    t = localtime(&timer);
    printf("출력 중입니다.");
    for (i; i < 3;i++)
    {
        printf("->");
        Sleep(300);
    }
    printf("출력 완료!");

    itoa(balance2, str2, 10);
    addCommas(str2); // 쉼표를 추가하여 형식화

    printf("\n잔액 조회: %s\n", str2);
    fprintf(fp, "\n잔액 조회: %s원\n거래시간: %d년 %d월 %d일 %d시 %d분 %d초\n", str2, t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec);
    printf("처리되었습니다.\n");
    N_time();
}

void N_time()
{
    time_t timer;
    struct tm* t;
    timer = time(NULL);
    t = localtime(&timer);
    printf("거래 시간: %d년 %d월 %d일 %d시 %d분 %d초\n",
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec);
}