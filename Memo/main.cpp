// 윈도우 api를 이용해 메모장 만들기 프로젝트

#include <stdlib.h>
#include <tchar.h>
#include <windows.h>

#define EM_SETINSERTMODE   (WM_USER + 97)
#define EM_GETINSERTMODE   (WM_USER + 98)

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

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
    WndClass.lpszMenuName = NULL;
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
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL | ES_OEMCONVERT, // ES_OEMCONVERT 스타일 추가
            0, 0, 100, 100, hWnd, NULL, g_hInst, NULL);
        break;
    case WM_SIZE:
        // 윈도우 크기가 변경될 때 에디트 컨트롤 크기 조정
        MoveWindow(g_hEdit, 0, 0, LOWORD(lParam), HIWORD(lParam), TRUE);
        break;
    case WM_COMMAND:
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
        ShowCaret(hWnd); // 캐럿 보이기
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