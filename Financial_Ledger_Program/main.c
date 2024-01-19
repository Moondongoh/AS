#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <limits.h>
#include "fun.h"
#include "fun_def.h"

int main()
{
    printf("*입력 가능 숫자 범위*\n\n");
    printf("MIN : %d\n", Limit_min);
    printf("MAX : %d\n\n", Limit_max);

    char assets[15];
    fp = fopen("list.txt", "w");
    if (fp == NULL)
    {
        printf("파일을 열 수 없습니다.\n");
        return 1;
    }

    printf("====== 가계부 =======\n");
    fixed();
    printf("초기 자산 내역을 적어주세요: ");
    scanf_s("%s", &assets, 15);
    while (1)
    {
        printf("현재 자산을 입력해주세요: ");
        if (scanf("%lld", &balance) != 1)
        {
            printf("잘못된 값을 입력 했습니다\n");
            while (getchar() != '\n');
            continue;
        }
        else if (balance < Limit_min)
        {
            printf("언더 플로우가 발생하였습니다. 다시 입력해주세요.\n");
            while (getchar() != '\n');
            continue;
        }
        else if (balance > Limit_max)
        {
            printf("오버 플로우가 발생하였습니다. 다시 입력해주세요.\n");
            while (getchar() != '\n');
            continue;
        }
        break;
    }
    balance2 = balance;
    balance2 = balance2 + Sum_income - Sum_expense;
    itoa(balance, str5, 10);
    addCommas(str5); // 쉼표를 추가하여 형식화
    itoa(balance2, str2, 10);
    addCommas(str2); // 쉼표를 추가하여 형식화
    fprintf(fp, "====================가계부====================\n내역: %s\n현재 자산: %s원\n고정 수입 지출 계산 후 자산: %s원\n", assets, str5, str2);

    while (1)
    {
        int choice;
        while (1)
        {
            printf("\n기능 선택 (1. 입금 2. 지출 3. 잔액 조회 4. 종료): ");
            if (scanf("%d", &choice) != 1)
            {
                system("cls");
                SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
                printf("잘못된 값을 입력 했습니다.\n");
                printf("이용하실 서비스를 선택해주세요.\n");
                while (getchar() != '\n');
                continue;
            }
            break;
        }

        switch (choice)
        {
        case 1:
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            plus();
            break;
        case 2:
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            minus();
            break;
        case 3:
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            print();
            break;
        case 4:
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            printf("프로그램을 종료합니다.\n");
            printf("이용해 주셔서 감사합니다.\n");
            fclose(fp);
            return 0;
        default:
            system("cls");
            SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
            printf("올바른 기능을 선택해주세요.\n");
            break;
        }
    }

    fclose(fp);
    return 0;
}