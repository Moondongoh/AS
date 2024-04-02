/*
[END]1. 윈도우 창을 생성한다.
[ING]>> 상단 툴바에 파일 서식 작업 크기가 있으며 각 기능들 추가 중이다.
[ING]1-1. 새로 만들기, 새 창, 저장 및 열기, 끝내기 기능 추가 할 예정
[ING]++ 툴바에 도형 칸 넣고 도형 그리기 추가 할 예정
[END]2. 마우스 좌 클릭 시 그리기 가능하다.
[END]3. 마우스 우 클릭 시 지우기 가능하다.
[END]4. 작업 영역(1920*1080)을 가지면 백 버퍼릉 만들어 더블 버퍼링 사용 중이다.
[ING]5. 스크롤 기능? 
[ING]6. 작업 영역에 격자 넣기?
*/
#include <windows.h>
#include "resource.h"

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

LPCTSTR lpszClass = TEXT("PAINT");

COLORREF LineColor = 0x00000000;
POINT startPoint = {0, 0};
POINT endPoint = {0, 0};
POINT prevEndPoint = {0, 0}; // 이전의 끝점을 저장하는 변수 추가
BOOL isDrawing = FALSE;
BOOL isDrawing_2 = FALSE;
BOOL isEraser = FALSE;
BOOL isRect = FALSE;
BOOL isEllipse = FALSE;

int cWidth; //지우개 크기 변수

// 백 버퍼 관련 변수
HDC hdcBuffer = NULL;
HBITMAP hBitmapBuffer = NULL;
HBITMAP hBitmapOld = NULL;
HBRUSH hWhiteBrush; // 백 버퍼를 하얀색으로 채우기 위한 브러시
RECT rect; // 백 버퍼 크기를 정의하는 사각형
//

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
	    
	// 메시지 루프가 종료되면, 백 버퍼 관련 리소스 해제
    if (hBitmapOld != NULL)
        SelectObject(hdcBuffer, hBitmapOld);
    if (hBitmapBuffer != NULL)
        DeleteObject(hBitmapBuffer);
    if (hdcBuffer != NULL)
        DeleteDC(hdcBuffer);
	//

    return (int)msg.wParam;
}

HDC hdc;

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;

    switch (iMsg)
    {
	case WM_CREATE:
		// 백 버퍼 초기화
        hdcBuffer = CreateCompatibleDC(NULL);
        hBitmapBuffer = CreateCompatibleBitmap(GetDC(hwnd), 1920, 1080);
        hBitmapOld = (HBITMAP)SelectObject(hdcBuffer, hBitmapBuffer);

		// 백 버퍼 전체를 하얀색으로 채우기
        hWhiteBrush = CreateSolidBrush(RGB(255, 255, 255));
        rect.left = 0;
        rect.top = 0;
        rect.right = 1920;
        rect.bottom = 1080;
        FillRect(hdcBuffer, &rect, hWhiteBrush);
        break;

    case WM_MOUSEMOVE:
        if (isDrawing || isEraser)
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

		case ID_FILE_PAINT_NEW:
			FillRect(hdcBuffer, &rect, hWhiteBrush);
			InvalidateRect(hwnd, NULL, TRUE);
			isDrawing = FALSE;
			isDrawing_2 = FALSE;
			isRect = FALSE;
			isEllipse = FALSE;
			break;

		case ID_FILE_NEW_PAINT_WINDOW:
			break;

		case ID_FILE_PAINT_OPEN:
			break;

		case ID_FILE_PAINT_SAVE:
			break;
			    
		case ID_FILE_EXIT:
			DestroyWindow(hwnd);
			break;

		case ID_Rectangle:
			isRect = TRUE;
			isDrawing = FALSE;
			isEraser = FALSE;
			break;

		case ID_Line:
			isDrawing_2 = TRUE;
			isRect = FALSE;
			isEllipse = FALSE;
			break;

		case ID_Ellipse:
			isEllipse = TRUE;
			isDrawing = FALSE;
			isEraser = FALSE;
			isRect =FALSE;
			break;

        }
        break;

    case WM_PAINT:
    
		{
			HDC hdc = BeginPaint(hwnd, &ps);
			

			// 백 버퍼에 그림을 그림
			HBRUSH hBrush;
			if (isEraser)
				hBrush = CreateSolidBrush(RGB(255, 255, 255)); // 지우개는 흰색으로 설정
			else
				hBrush = CreateSolidBrush(LineColor);
			SelectObject(hdcBuffer, hBrush);
        
			if (isDrawing || isEraser || isRect || isEllipse || isDrawing_2)
			{
				if (isEraser)
					SelectObject(hdcBuffer, CreatePen(PS_SOLID, cWidth, RGB(255, 255, 255))); // 펜 없이 그리기
				else if(isRect)
					Rectangle(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);
				else if(isEllipse)
					Ellipse(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);
				else if(isDrawing_2)
				{
					isDrawing = TRUE;
				}
				else
					SelectObject(hdcBuffer, CreatePen(PS_SOLID, cWidth, LineColor)); // 펜 선택
				MoveToEx(hdcBuffer, prevEndPoint.x, prevEndPoint.y, NULL);
				LineTo(hdcBuffer, endPoint.x, endPoint.y);
			}
			DeleteObject(hBrush);

			// 백 버퍼에 그린 그림을 화면에 복사
			BitBlt(hdc, 0, 0, 1920, 1080, hdcBuffer, 0, 0, SRCCOPY);

			EndPaint(hwnd, &ps);
		}
		break;


    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    }

    return DefWindowProc(hwnd, iMsg, wParam, lParam);
}
