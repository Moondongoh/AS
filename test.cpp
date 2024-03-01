// 윈도우 api를 이용해 메모장 만들기 프로젝트

/*
1. 영어 및 한글을 입력 받을 수 있어야함.
2. Home, End, Ins, Del, Page up,down기능
3. 메모장 툴바 제작 >> 파일, 편집, 서식
   -각 메뉴에 기능들 추가 (새로 만들기, 새창 등등...)
4. Accelerator(단축키)추가
5. 저장 및 열기 기능
6. 동적할당으로 메모리 배정
7. 각 함수의 이해 및 기본으로 사용 할 수 있는 기능 파악
*/

/*충격 사실 에디터 컨트롤을 사용하면 대강 텍스트 입력 후 Home, End, Insert, Delete, Page Up, Page Down 기능을 사용 할 수 있다.
하지만 일단은 에디터 컨트롤을 이용해서 텍스트는 입력을 받고 뒷 기능들을 내가 직접 추가해서 사용 할 수 있도록 하기로함.
*/

/*

2월 26일 메모장 리소스 파일 생성 툴바 제작 -파일(F)칸 생성 
		 수평, 수직 스크롤바 생성
		 새로 만들기 및 새창 생성

2월 27일 편집(E), 서식(O)칸 생성 및 단축키 추가
		 새 창을 열게 되면 원래는 그냥 에디터 컨트롤만 적용된 창이 생성됨.
		 이제는 툴바도 포함된 창을 보여줌
		 실행창 크기에 따라 스크롤바 수정하도록

2월 28일 
		엑셀레이터(단축키) 설정
		키 다운 기능으로 Home, End, Delete, Insert, Page Up, Down 작동하도록 작업 완료
		열기 및 저장 방법 찾기

*/
#include <stdlib.h>
#include <tchar.h>
#include <windows.h>
#include<limits.h>
#include"resource.h"

#define EM_SETINSERTMODE   (WM_USER + 97)
#define EM_GETINSERTMODE   (WM_USER + 98)

#define IDR_MENU1						101
#define ID_FILE_새로_만들기(N)			40009
#define ID_FILE_새_창(W)				40010
#define ID_FILE_열기(O)					40011
#define ID_FILE_저장(S)					40012
#define ID_FILE_다른 이름으로 저장(A)	40013
#define ID_FILE_끝내기(X)				40014

#define ID_FILE_실행_취소(Z)			40017
#define ID_FILE_잘라내기(T)				40018
#define ID_FILE_복사(C)					40019
#define ID_FILE_붙여넣기(P)				40020
#define ID_FILE_삭제(L)					40021
#define ID_FILE_모두_선택(A)			40023


LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

void CreateNewWindow(); // 새 창 열기 함수

HINSTANCE g_hinst;
LPCTSTR lpszClass=TEXT("Menu"); // 메뉴바 만들때 사용함

HINSTANCE g_hInst;

HWND g_hEdit; // 에디트 컨트롤 핸들

int WINAPI WinMain(HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR lpszCmdParam,
    int nCmdShow)
{
    HWND hWnd;
    MSG Message;
    WNDCLASS WndClass;
    g_hInst = hInstance;

    // 윈도우 클래스 초기화
    WndClass.cbClsExtra = 0;
    WndClass.cbWndExtra = 0;
    WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    WndClass.hInstance = hInstance;
    WndClass.lpfnWndProc = (WNDPROC)WndProc;
    WndClass.lpszClassName = _T("야야문문동동오오메메모모장장");
	WndClass.lpszMenuName=MAKEINTRESOURCE(IDR_MENU1);
    WndClass.style = CS_HREDRAW | CS_VREDRAW;

    // 윈도우 클래스 생성
    RegisterClass(&WndClass);

    // 윈도우 객체 생성
    hWnd = CreateWindow(_T("야야문문동동오오메메모모장장"),
        _T("야야문문동동오오메메모모장장"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        NULL,
        (HMENU)NULL,
        hInstance,
        NULL);

    // 윈도우 창 띄우기
    ShowWindow(hWnd, nCmdShow);

    while (GetMessage(&Message, 0, 0, 0))
    {
        TranslateMessage(&Message);
        DispatchMessage(&Message);
    }

    return Message.wParam;
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT iMessage, WPARAM wParam, LPARAM lParam)
{
    switch (iMessage)
    {

	/* 현재 WM_CTEATE가 에디트 컨트롤을 생성함.*/

    case WM_CREATE:
        // 에디트 컨트롤 생성
        g_hEdit = CreateWindow(_T("edit"), NULL,
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL | WS_HSCROLL | WS_VSCROLL, // ES_OEMCONVERT 스타일 추가
            0, 0, 100, 100, hWnd, NULL, g_hInst, NULL);
        break;
	
	/* 창의 크기에 따라서 텍스트 창과 스크롤의 크기를 조정해줌.*/
    case WM_SIZE:
        // 윈도우 크기가 변경될 때 에디트 컨트롤 크기 조정
        MoveWindow(g_hEdit, 0	, 0, LOWORD(lParam), HIWORD(lParam), TRUE);
        break;


	case WM_COMMAND:
    switch (LOWORD(wParam))
    {
    case ID_FILE_새로_만들기(N):
        // 새로 만들기 메뉴 아이템이 선택되었을 때 처리할 코드
        SendMessage(g_hEdit, WM_SETTEXT, 0, (LPARAM)TEXT(""));
        break;

    case ID_FILE_새_창(W):
        // 새 창 열기 메뉴 아이템이 선택되었을 때 처리할 코드
        CreateNewWindow();
        break;

    case ID_FILE_실행_취소(Z):
        // 실행 취소 메뉴 아이템이 선택되었을 때 처리할 코드
        SendMessage(g_hEdit, EM_UNDO, 0, 0);
        break;

    case ID_FILE_잘라내기(T):
        // 잘라내기 메뉴 아이템이 선택되었을 때 처리할 코드
        SendMessage(g_hEdit, WM_CUT, 0, 0);
        break;

    case ID_FILE_복사(C):
        // 복사 메뉴 아이템이 선택되었을 때 처리할 코드
        SendMessage(g_hEdit, WM_COPY, 0, 0);
        break;

    case ID_FILE_붙여넣기(P):
        // 붙여넣기 메뉴 아이템이 선택되었을 때 처리할 코드
        SendMessage(g_hEdit, WM_PASTE, 0, 0);
        break;

    case ID_FILE_삭제(D):
        // 삭제 메뉴 아이템이 선택되었을 때 처리할 코드
        SendMessage(g_hEdit, WM_CLEAR, 0, 0);
        break;

    case ID_FILE_모두_선택(A):
        // 모두 선택 메뉴 아이템이 선택되었을 때 처리할 코드
        SendMessage(g_hEdit, EM_SETSEL, 0, -1);
        break;

    default:
        return DefWindowProc(hWnd, iMessage, wParam, lParam);
    }
    break;


        // 에디트 컨트롤에서 이벤트 발생 시 처리
        // 여기서 텍스트 입력 처리 등을 수행할 수 있음
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    //case WM_SETFOCUS:
    //{
    //    HDC hdc = GetDC(hWnd); // hdc 선언 추가
    //    CreateCaret(hWnd, NULL, 2, GetSystemMetrics(SM_CYCURSOR)); // 캐럿 생성, 삽입 모드로 변경
    //    //ShowCaret(hWnd); // 캐럿 보이기
    //    SetCaretPos(0, 0); // 캐럿의 위치 설정
    //
    //    // 현재 입력 모드가 "덮어쓰기" 모드인 경우 삽입 모드로 변경
    //    if (!(SendMessage(g_hEdit, EM_GETINSERTMODE, 0, 0)))
    //        SendMessage(g_hEdit, EM_SETINSERTMODE, TRUE, 0);

    //    ReleaseDC(hWnd, hdc); // hdc 릴리스
    //    break;
    //}

    case WM_KILLFOCUS:
        HideCaret(hWnd); // 캐럿 숨기기
        DestroyCaret(); // 캐럿 파괴
        break;

	case WM_KEYDOWN:
	{
		// 에디트 컨트롤에서 키보드 이벤트를 처리
		switch (wParam)
		{
		case VK_HOME: // 홈 키
			SendMessage(g_hEdit, EM_SETSEL, 0, 0);
			break;

		case VK_END: // 엔드 키
			SendMessage(g_hEdit, EM_SETSEL, -1, -1);
			break;

		case VK_DELETE: // 삭제 키
			SendMessage(g_hEdit, WM_CLEAR, 0, 0);
			break;

		case VK_INSERT: // 인서트 키
			{
			 // 현재의 덮어쓰기 모드를 가져옴
				BOOL bInsertMode = SendMessage(g_hEdit, EM_GETINSERTMODE, 0, 0);
				// 덮어쓰기 모드를 토글하여 설정
				SendMessage(g_hEdit, EM_SETINSERTMODE, !bInsertMode, 0);
			 }
			break;
			
		case VK_PRIOR: // Page Up 키
			SendMessage(g_hEdit, EM_SCROLL, SB_PAGEUP, 0);
			break;

		case VK_NEXT: // Page Down 키
			SendMessage(g_hEdit, EM_SCROLL, SB_PAGEDOWN, 0);
			break;
		}
	}
	break;

    default:
        return DefWindowProc(hWnd, iMessage, wParam, lParam);
    }
    return 0;
}

// 새 창 열기 함수
void CreateNewWindow() 
{
    HWND hWndNew;
    MSG Message;
    WNDCLASS WndClass;

    // 현재 창과 동일한 메뉴를 가지도록 설정
    WndClass.cbClsExtra = 0;
    WndClass.cbWndExtra = 0;
    WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    WndClass.hInstance = g_hInst;
    WndClass.lpfnWndProc = (WNDPROC)WndProc;
    WndClass.lpszClassName = _T("NewWindow");
    WndClass.lpszMenuName = MAKEINTRESOURCE(IDR_MENU1); // 현재 창과 동일한 메뉴 리소스 ID로 설정
    WndClass.style = CS_HREDRAW | CS_VREDRAW;

    RegisterClass(&WndClass);

    hWndNew = CreateWindow(_T("NewWindow"),
        _T("야야문문동동오오메메모모장장새새창창"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        NULL,
        NULL,
        g_hInst,
        NULL);

    ShowWindow(hWndNew, SW_NORMAL);
    UpdateWindow(hWndNew);

    while (GetMessage(&Message, 0, 0, 0)) 
	{
        TranslateMessage(&Message);
        DispatchMessage(&Message);
    }
}

