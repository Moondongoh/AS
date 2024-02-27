// 윈도우 api를 이용해 메모장 만들기 프로젝트

/*

2월 26일 메모장 리소스 파일 생성 툴바 제작 -파일(F)칸 생성 
		 수평, 수직 스크롤바 생성
		 새로 만들기 및 새창 생성

*/
#include <stdlib.h>
#include <tchar.h>
#include <windows.h>
#include"resource.h"

#define MAX 
#define EM_SETINSERTMODE   (WM_USER + 97)
#define EM_GETINSERTMODE   (WM_USER + 98)

#define IDR_MENU1						101
#define ID_FILE_새로_만들기(N)			40009
#define ID_FILE_새_창(W)				40010
#define ID_FILE_열기(O)					40011
#define ID_FILE_저장(S)					40012
#define ID_FILE_다른 이름으로 저장(A)	40013
#define ID_FILE_끝내기(X)				40014

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

void CreateNewWindow();

HINSTANCE g_hinst;
LPCTSTR lpszClass=TEXT("Menu");

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
    case WM_CREATE:
        // 에디트 컨트롤 생성
        g_hEdit = CreateWindow(_T("edit"), NULL,
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL | ES_OEMCONVERT | WS_HSCROLL | WS_VSCROLL, // ES_OEMCONVERT 스타일 추가
            0, 0, 100, 100, hWnd, NULL, g_hInst, NULL);
        break;
    case WM_SIZE:
        // 윈도우 크기가 변경될 때 에디트 컨트롤 크기 조정
        MoveWindow(g_hEdit, 0	, 0, LOWORD(lParam), HIWORD(lParam), TRUE);
        break;
    case WM_COMMAND:
		switch (LOWORD(wParam))
		{
		case ID_FILE_새로_만들기(N):
        // 새로 만들기 메뉴 아이템이 선택되었을 때 처리할 코드를 여기에 추가하세요.
        // 새 파일을 만들기 전에 기존 내용을 저장할 것인지 확인하는 등의 작업을 수행할 수 있습니다.
        // 이 예제에서는 단순히 에디트 컨트롤을 비웁니다.
        SendMessage(g_hEdit, WM_SETTEXT, 0, (LPARAM)TEXT(""));
        break;
		case ID_FILE_새_창(W): //새창
	    CreateNewWindow();
		break;
		// 다른 메뉴 아이템들에 대한 처리도 추가할 수 있습니다.
		}
		break;
        // 에디트 컨트롤에서 이벤트 발생 시 처리
        // 여기서 텍스트 입력 처리 등을 수행할 수 있음
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_SETFOCUS:
    {
        HDC hdc = GetDC(hWnd); // hdc 선언 추가
        CreateCaret(hWnd, NULL, 2, GetSystemMetrics(SM_CYCURSOR)); // 캐럿 생성, 삽입 모드로 변경
        //ShowCaret(hWnd); // 캐럿 보이기
        SetCaretPos(0, 0); // 캐럿의 위치 설정
    
        // 현재 입력 모드가 "덮어쓰기" 모드인 경우 삽입 모드로 변경
        if (!(SendMessage(g_hEdit, EM_GETINSERTMODE, 0, 0)))
            SendMessage(g_hEdit, EM_SETINSERTMODE, TRUE, 0);

        ReleaseDC(hWnd, hdc); // hdc 릴리스
        break;
    }
    case WM_KILLFOCUS:
        HideCaret(hWnd); // 캐럿 숨기기
        DestroyCaret(); // 캐럿 파괴
        break;
    case WM_KEYDOWN:
        // 키 다운 메시지를 처리하여 Insert 키를 눌렀을 때 덮어쓰기 모드를 토글
        if (wParam == VK_INSERT)
        {
            BOOL bInsertMode = SendMessage(g_hEdit, EM_GETINSERTMODE, 0, 0);
            SendMessage(g_hEdit, EM_SETINSERTMODE, !bInsertMode, 0);
        }
        break;
    default:
        return DefWindowProc(hWnd, iMessage, wParam, lParam);
    }
    return 0;
}

void CreateNewWindow() {
    HWND hWndNew;
    MSG Message;
    WNDCLASS WndClass;

    WndClass.cbClsExtra = 0;
    WndClass.cbWndExtra = 0;
    WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    WndClass.hInstance = g_hInst;
    WndClass.lpfnWndProc = (WNDPROC)WndProc;
    WndClass.lpszClassName = _T("NewWindow");
    WndClass.lpszMenuName = NULL;
    WndClass.style = CS_HREDRAW | CS_VREDRAW;

    RegisterClass(&WndClass);

    hWndNew = CreateWindow(_T("NewWindow"),
        _T("새 창"),
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

    while (GetMessage(&Message, 0, 0, 0)) {
        TranslateMessage(&Message);
        DispatchMessage(&Message);
    }
}
