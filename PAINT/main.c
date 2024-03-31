#include <windows.h>
#include "resource.h"

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

LPCTSTR lpszClass = TEXT("PAINT");

COLORREF LineColor = 0x00000000;
POINT startPoint = {0, 0};
POINT endPoint = {0, 0};
POINT prevEndPoint = {0, 0}; // 이전의 끝점을 저장하는 변수 추가
BOOL isDrawing = FALSE;
BOOL isEraser = FALSE;

int cWidth; //지우개 크기 변수

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{

    HWND hwnd;
    MSG msg;
    WNDCLASS WndClass;
    WndClass.style = CS_HREDRAW | CS_VREDRAW;
    WndClass.lpfnWndProc = WndProc;
    WndClass.cbClsExtra = 0;
    WndClass.cbWndExtra = 0;
    WndClass.hInstance = hInstance;
    WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    WndClass.lpszMenuName = NULL;
    WndClass.lpszClassName = lpszClass;
    RegisterClass(&WndClass);

    hwnd = CreateWindow(lpszClass,
                        lpszClass,
                        WS_OVERLAPPEDWINDOW,
                        100,
                        50,
                        600,
                        400,
                        NULL,
                        LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)),
                        hInstance,
                        NULL);
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int)msg.wParam;
}

HDC hdc;

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;

    switch (iMsg)
    {
	case WM_CREATE:
        break;

    case WM_MOUSEMOVE:
        if (isDrawing)
        {
            prevEndPoint = endPoint; // 이전 끝점을 현재 끝점으로 갱신
            endPoint.x = LOWORD(lParam);
            endPoint.y = HIWORD(lParam);
            InvalidateRect(hwnd, NULL, FALSE);
        }
		if (isEraser)
		{
            prevEndPoint = endPoint; // 이전 끝점을 현재 끝점으로 갱신
            endPoint.x = LOWORD(lParam);
            endPoint.y = HIWORD(lParam);
            InvalidateRect(hwnd, NULL, FALSE);
		}
        break;

		//LEFT : 그리기 RIGHT : 지우기
    case WM_LBUTTONDOWN:
        startPoint.x = endPoint.x = LOWORD(lParam);
        startPoint.y = endPoint.y = HIWORD(lParam);
        prevEndPoint = startPoint; // 이전 끝점을 현재 시작점으로 설정
        isDrawing = TRUE;
        break;

    case WM_LBUTTONUP:
        isDrawing = FALSE;
        break;

	    
	case WM_RBUTTONDOWN:
		isEraser = TRUE;
		startPoint.x = endPoint.x = LOWORD(lParam);
        startPoint.y = endPoint.y = HIWORD(lParam);
        prevEndPoint = startPoint; // 이전 끝점을 현재 시작점으로 설정
        break;

    case WM_RBUTTONUP:
		isEraser = FALSE;
        break;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        // 선 색상
        case ID_RED:
            LineColor = RGB(255, 0, 0);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_BLUE:
            LineColor = RGB(0, 0, 255);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_YELLOW:
            LineColor = RGB(255, 255, 0);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_GREEN:
            LineColor = RGB(0, 255, 0);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_BLACK:
            LineColor = RGB(0, 0, 0);
            InvalidateRect(hwnd, NULL, FALSE);
            break;

        // 이 지우개 사이즈  
		case ID_SIZE_10:
            cWidth = 10;
            InvalidateRect(hwnd, NULL, FALSE);
            break;
					
		case ID_SIZE_20:
            cWidth = 20;
            InvalidateRect(hwnd, NULL, FALSE);
            break;
					
		case ID_SIZE_30:
            cWidth = 30;
            InvalidateRect(hwnd, NULL, FALSE);
            break;
        }
        break;

    case WM_PAINT:
        hdc = BeginPaint(hwnd, &ps);

        if (isDrawing)
        {
            HPEN hPen = CreatePen(PS_SOLID, 1, LineColor);
            SelectObject(hdc, hPen);
            MoveToEx(hdc, prevEndPoint.x, prevEndPoint.y, NULL); // 이전 끝점부터 시작
            LineTo(hdc, endPoint.x, endPoint.y);
            DeleteObject(hPen);
        }

		if(isEraser)
		{
            HPEN hPen = CreatePen(PS_SOLID, cWidth, RGB(255,255,255));
            SelectObject(hdc, hPen);
            MoveToEx(hdc, prevEndPoint.x, prevEndPoint.y, NULL); // 이전 끝점부터 시작
            LineTo(hdc, endPoint.x, endPoint.y);
            DeleteObject(hPen);
		}

        EndPaint(hwnd, &ps);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    }

    return DefWindowProc(hwnd, iMsg, wParam, lParam);
}
