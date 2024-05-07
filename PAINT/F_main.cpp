/*
[END]1. 윈도우 창을 생성한다.
[END]>> 상단 툴바에 파일 서식 작업 크기가 있으며 각 기능을 사용 할 수 있다.
[END]1-1. 새로 만들기, 새 창, 저장 및 열기, 끝내기을 사용 할 수 있다.
[END]++ 툴바에 도형 칸 넣고 도형 그릴 수 있다.
[END]++ 선의 굵기, 선의 색상, 도형 색 채우기를 할 수 있다.
[END]2. 마우스 좌 클릭 시 그리기 가능하다.
[END]3. 마우스 우 클릭 시 지우기 가능하다.
[END]4. 작업 영역(1920*1080)을 가지면 백 버퍼를 만들어 더블 버퍼링 사용 중이다.
[NO]5. 스크롤 기능? 
[NO]6. 작업 영역에 격자 넣기?
*/
#include <windows.h>
#include <windef.h>
#include "resource.h"

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

void CreateNewPaintWindow(HINSTANCE hInstance);
bool trySave(HWND hwnd);
bool tryOpen(HWND hwnd);

LPCTSTR lpszClass = TEXT("PAINT");

COLORREF LineColor = 0x00000000;
COLORREF FullColor = 0x00000000;

POINT startPoint = {0, 0};
POINT endPoint = {0, 0};
POINT prevEndPoint = {0, 0};	// 이전의 끝점을 저장하는 변수 추가

BOOL isDrawing = FALSE;			// 기본 그리기
BOOL isDrawing_2 = FALSE;		// 메뉴바 그리기
BOOL isEraser = FALSE;			// 지우기
BOOL isRect = FALSE;			// 사각형
BOOL isEllipse = FALSE;			// 타원
BOOL isRed = FALSE;				// 빨간색
BOOL isBlue = FALSE;			// 파란색
BOOL isYellow = FALSE;			// 노란색
BOOL isGreen = FALSE;			// 초록색
BOOL isBlack= FALSE;			// 검정색
BOOL isZero= FALSE;				// 투명색
BOOL is_F_Red = FALSE;			// 빨간색
BOOL is_F_Blue = FALSE;			// 파란색
BOOL is_F_Yellow = FALSE;		// 노란색
BOOL is_F_Green = FALSE;		// 초록색
BOOL is_F_Black= FALSE;			// 검정색
BOOL is_F_Zero= FALSE;			// 투명색


int cWidth;						//지우개 크기 변수

// 백 버퍼 관련 변수
HDC hdcBuffer = NULL;
HBITMAP hBitmapBuffer = NULL;
HBITMAP hBitmapOld = NULL;
HBRUSH hWhiteBrush;				// 백 버퍼를 하얀색으로 채우기 위한 브러시
RECT rect;						// 백 버퍼 크기를 정의하는 사각형

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
                        CW_USEDEFAULT,
                        CW_USEDEFAULT,
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
    static HBITMAP hTempBitmap = NULL;								// 임시 백 버퍼를 저장하기 위한 변수
    static HDC hdcTemp = NULL;										// 임시 백 버퍼의 DC를 저장하기 위한 변수

    PAINTSTRUCT ps;
	HBITMAP hOldTempBitmap = NULL;

    switch (iMsg)
    {
    case WM_CREATE:

		// Window창을 화면 정중앙에 위치 시키는 코드
		int x, y, width, height;
		RECT rtDesk, rtWindow;
		GetWindowRect(GetDesktopWindow(), &rtDesk);
		GetWindowRect(hwnd, &rtWindow);

		width = rtWindow.right - rtWindow.left;
		height = rtWindow.bottom - rtWindow.top;

		x = (rtDesk.right - width) / 2;
		y = (rtDesk.bottom - height) / 2;

		MoveWindow(hwnd, x, y, width, height, TRUE);
		//

        // 백 버퍼 초기화
        hdcBuffer = CreateCompatibleDC(NULL);
        hBitmapBuffer = CreateCompatibleBitmap(GetDC(hwnd), 1920, 1080);
        hBitmapOld = (HBITMAP)SelectObject(hdcBuffer, hBitmapBuffer);
        hWhiteBrush = CreateSolidBrush(RGB(255, 255, 255));         // 백 버퍼 전체를 하얀색으로 채우기

        rect.left = 0;
        rect.top = 0;
        rect.right = 1920;
        rect.bottom = 1080;

        FillRect(hdcBuffer, &rect, hWhiteBrush);

        isDrawing = FALSE;
        isDrawing_2 = FALSE; 
        isEraser = FALSE;
        isRect = FALSE;
        isEllipse = FALSE;

        break;
		/*

		*/
    case WM_MOUSEMOVE:
    if (wParam & MK_LBUTTON || wParam & MK_RBUTTON ) 
	{
        if (isDrawing || isEraser) {
            prevEndPoint = endPoint;
            endPoint.x = LOWORD(lParam);
            endPoint.y = HIWORD(lParam);
            InvalidateRect(hwnd, NULL, FALSE);
        }
        else if (isRect || isEllipse) {
            endPoint.x = LOWORD(lParam);
            endPoint.y = HIWORD(lParam);

            // 이전에 그린 도형을 지우기 위해 화면을 백 버퍼로 복원
            BitBlt(hdcBuffer, 0, 0, 1920, 1080, hdcTemp, 0, 0, SRCCOPY);

            // 새로운 위치에 도형 그리기
            SelectObject(hdcBuffer, (HBRUSH)GetStockObject(NULL_BRUSH));
            /*if (isRect)
                Rectangle(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);
            else if (isEllipse)
                Ellipse(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);*/

            InvalidateRect(hwnd, NULL, FALSE);
        }
    }
    break;

	case WM_LBUTTONDOWN:

    if (isRect || isEllipse) {
        // 임시 백 버퍼 생성 및 초기화
        hdcTemp = CreateCompatibleDC(hdcBuffer);
        HBITMAP hTempBitmap = CreateCompatibleBitmap(hdcBuffer, 1920, 1080);
        HBITMAP hOldTempBitmap = (HBITMAP)SelectObject(hdcTemp, hTempBitmap);
        BitBlt(hdcTemp, 0, 0, 1920, 1080, hdcBuffer, 0, 0, SRCCOPY);

        startPoint.x = endPoint.x = LOWORD(lParam);
        startPoint.y = endPoint.y = HIWORD(lParam);
    }
    else 
	{
        startPoint.x = endPoint.x = LOWORD(lParam);
        startPoint.y = endPoint.y = HIWORD(lParam);
        prevEndPoint = startPoint;
        isDrawing = TRUE;
    }
    break;

	case WM_LBUTTONUP:
		if (isRect || isEllipse) {
			// 최종 도형 그리기
			SelectObject(hdcBuffer, (HBRUSH)GetStockObject(NULL_BRUSH));
			if (isRect)
				Rectangle(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);
			else if (isEllipse)
				Ellipse(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);

			// 임시 백 버퍼 해제
			SelectObject(hdcTemp, hOldTempBitmap);
			DeleteObject(hTempBitmap);
			DeleteDC(hdcTemp);
        
			InvalidateRect(hwnd, NULL, FALSE);
		}
		break;

    case WM_RBUTTONDOWN:
        // 지우개 모드로 설정
        isEraser = TRUE;
		isRect = FALSE;
		isEllipse = FALSE;
		isDrawing = TRUE;
        startPoint.x = endPoint.x = LOWORD(lParam);
        startPoint.y = endPoint.y = HIWORD(lParam);
        prevEndPoint = startPoint; // 이전 끝점을 현재 시작점으로 설정
        break;

    case WM_RBUTTONUP:
        // 지우개 모드 해제
        isEraser = FALSE;
        break;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        // 선 색상
        case ID_RED:
            LineColor = RGB(255, 0, 0);
			isRed = TRUE;
            //InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_BLUE:
            LineColor = RGB(0, 0, 255);
			isBlue = TRUE;
            //InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_YELLOW:
            LineColor = RGB(255, 255, 0);
			isYellow = TRUE;
            //InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_GREEN:
            LineColor = RGB(0, 255, 0);
			isGreen = TRUE;
            //InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_BLACK:
            LineColor = RGB(0, 0, 0);
			isBlack = TRUE;
            //InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_Full_Red:
            FullColor = RGB(255, 0, 0);
            is_F_Red = TRUE;
            break;

        case ID_Full_Blue:
            FullColor = RGB(0, 0, 255);
            is_F_Blue = TRUE;
            break;

        case ID_Full_Yellow:
            FullColor = RGB(255, 255, 0);
            is_F_Yellow = TRUE;
            break;

        case ID_Full_Green:
            FullColor = RGB(0, 255, 0);
            is_F_Green = TRUE;
            break;

        case ID_Full_Black:
            FullColor = RGB(0, 0, 0);
            is_F_Black = TRUE;
            break;

        case ID_Full_Zero:
            is_F_Red = FALSE;
            is_F_Blue =FALSE;
            is_F_Yellow =FALSE;
            is_F_Green =FALSE;
            is_F_Black =FALSE;
			is_F_Zero = TRUE;
            break;

        // 작업 사이즈  
        case ID_SIZE_10:
            cWidth = 10;
            //InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_SIZE_20:
            cWidth = 20;
            //InvalidateRect(hwnd, NULL, FALSE);
            break;

        case ID_SIZE_30:
            cWidth = 30;
            //InvalidateRect(hwnd, NULL, FALSE);
            break;

		// 새로 만들기
        case ID_FILE_PAINT_NEW:
            FillRect(hdcBuffer, &rect, hWhiteBrush);
            InvalidateRect(hwnd, NULL, TRUE);
            isDrawing = FALSE;
            isDrawing_2 = FALSE;
            isRect = FALSE;
            isEllipse = FALSE;
            break;
		
		// 새 창 열기
        case ID_FILE_NEW_PAINT_WINDOW:
            CreateNewPaintWindow(GetModuleHandle(NULL));
            break;
		
		// 열기
        case ID_FILE_NEW_PAINT_OPEN:
			tryOpen(hwnd);
            break;

		// 저장
        case ID_FILE_NEW_PAINT_SAVE:
            trySave(hwnd);
			break;

        case ID_FILE_EXIT:
            DestroyWindow(hwnd);
            break;

		// 메뉴바 사각형 그리기
        case ID_Rectangle:
            isRect = TRUE;
            isDrawing = FALSE;
            isDrawing_2= FALSE;
            isEraser = FALSE;
            isEllipse = FALSE;
            break;

		// 메뉴바 선 그리기
        case ID_Line:
            isDrawing_2 = TRUE;
            isRect = FALSE;
            isEllipse = FALSE;
            break;

		// 메뉴바 원 그리기
        case ID_Ellipse:
            isEllipse = TRUE;
            isDrawing = FALSE;
			isDrawing_2= FALSE;
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
            HPEN hPen;
            if (isEraser)
            {
                hBrush = CreateSolidBrush(RGB(255, 255, 255)); // 지우개는 흰색으로 설정
                hPen = CreatePen(PS_SOLID, cWidth, RGB(255, 255, 255));
            }
            else
            {
                if (is_F_Red || is_F_Blue || is_F_Yellow || is_F_Green || is_F_Black)
                {
                    hBrush = CreateSolidBrush(FullColor);
                }
                else
                {
                    hBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
                }

                if (isRed || isBlue || isYellow || isGreen || isBlack)
                {
                    hPen = CreatePen(PS_SOLID, cWidth, LineColor);
                }
                else
                {
                    hPen = CreatePen(PS_SOLID, cWidth, RGB(0, 0, 0)); // 기본 검정색 펜
                }
            }
            SelectObject(hdcBuffer, hBrush);
            SelectObject(hdcBuffer, hPen);
            if (isEllipse)
            {
                isDrawing = FALSE;
                Ellipse(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);
                isEllipse = TRUE;
            }

            if (isRect)
            {
                Rectangle(hdcBuffer, startPoint.x, startPoint.y, endPoint.x, endPoint.y);
                isRect = TRUE;
            }

            else if (isDrawing || isEraser || isDrawing_2)
            {
                MoveToEx(hdcBuffer, prevEndPoint.x, prevEndPoint.y, NULL);
                LineTo(hdcBuffer, endPoint.x, endPoint.y);
            }
            DeleteObject(hBrush);
            DeleteObject(hPen);
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

void CreateNewPaintWindow(HINSTANCE hInstance)
{
    HWND newHwnd;
    MSG newMsg;
    WNDCLASS newWndClass;

    // 새로운 윈도우 클래스를 등록합니다.
    newWndClass.style = CS_HREDRAW | CS_VREDRAW;
    newWndClass.lpfnWndProc = WndProc;
    newWndClass.cbClsExtra = 0;
    newWndClass.cbWndExtra = 0;
    newWndClass.hInstance = hInstance;
    newWndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    newWndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    newWndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    newWndClass.lpszMenuName = NULL;
    newWndClass.lpszClassName = TEXT("NEW_PAINT_WINDOW");
    RegisterClass(&newWndClass);

    // 새로운 윈도우를 생성합니다.
    newHwnd = CreateWindow(TEXT("NEW_PAINT_WINDOW"),
                            TEXT("New Paint Window"),
                            WS_OVERLAPPEDWINDOW,
                            CW_USEDEFAULT,
                            CW_USEDEFAULT,
                            600,
                            400,
                            NULL,
                            LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)),
                            hInstance,
                            NULL);
    ShowWindow(newHwnd, SW_SHOW);
    UpdateWindow(newHwnd);

    // 새로운 윈도우의 메시지 루프를 실행합니다.
    while (GetMessage(&newMsg, NULL, 0, 0))
    {
        TranslateMessage(&newMsg);
        DispatchMessage(&newMsg);
    }
}

bool trySave(HWND hwnd) {
  OPENFILENAME OfnData;
  ZeroMemory(&OfnData, sizeof(OfnData));
  OfnData.lStructSize = sizeof(OfnData);
  OfnData.hwndOwner = hwnd;
  OfnData.lpstrFilter = L"Bitmap Files (*.bmp)\0*.bmp\0All Files (*.*)\0*.*\0";
  OfnData.nFilterIndex = 1;
  OfnData.lpstrFile = NULL;
  OfnData.nMaxFile = 0;
  OfnData.lpstrDefExt = L"bmp";
  OfnData.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;

  TCHAR FileName[MAX_PATH] = L"";
  OfnData.lpstrFile = FileName;
  OfnData.nMaxFile = MAX_PATH;

  if (GetSaveFileName(&OfnData) == TRUE) {
    RECT ClientRect;
    GetClientRect(hwnd, &ClientRect);
    int Width = ClientRect.right - ClientRect.left;
    int Height = ClientRect.bottom - ClientRect.top;

    HDC ScreenDC = GetDC(hwnd);
    HDC MemDC = CreateCompatibleDC(ScreenDC);
    HBITMAP MemBitmap = CreateCompatibleBitmap(ScreenDC, Width, Height);
    HBITMAP OldBitmap = (HBITMAP)SelectObject(MemDC, MemBitmap);

    BitBlt(MemDC, 0, 0, Width, Height, ScreenDC, 0, 0, SRCCOPY);

    BITMAPFILEHEADER BfHeader;
    BITMAPINFOHEADER BiHeader;
    BITMAP Bitmap;

    GetObject(MemBitmap, sizeof(BITMAP), &Bitmap);

    BfHeader.bfType = 0x4D42; // "BM"
    BfHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    BfHeader.bfSize =
        BfHeader.bfOffBits + Bitmap.bmWidthBytes * Bitmap.bmHeight;
    BfHeader.bfReserved1 = 0;
    BfHeader.bfReserved2 = 0;

    BiHeader.biSize = sizeof(BITMAPINFOHEADER);
    BiHeader.biWidth = Bitmap.bmWidth;
    BiHeader.biHeight = Bitmap.bmHeight;
    BiHeader.biPlanes = 1;
    BiHeader.biBitCount = Bitmap.bmBitsPixel;
    BiHeader.biCompression = BI_RGB;
    BiHeader.biSizeImage = Bitmap.bmWidthBytes * Bitmap.bmHeight;
    BiHeader.biXPelsPerMeter = 0;
    BiHeader.biYPelsPerMeter = 0;
    BiHeader.biClrUsed = 0;
    BiHeader.biClrImportant = 0;

    HANDLE FileHandle = CreateFile(FileName, GENERIC_WRITE, 0, NULL,
                                   CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (FileHandle != INVALID_HANDLE_VALUE) {
      DWORD BytesWritten = 0;
      WriteFile(FileHandle, &BfHeader, sizeof(BITMAPFILEHEADER), &BytesWritten,
                NULL);
      WriteFile(FileHandle, &BiHeader, sizeof(BITMAPINFOHEADER), &BytesWritten,
                NULL);

      BYTE *BitmapBits = new BYTE[BiHeader.biSizeImage];
      if (GetDIBits(MemDC, MemBitmap, 0, BiHeader.biHeight, BitmapBits,
                    (BITMAPINFO *)&BiHeader, DIB_RGB_COLORS)) {
        WriteFile(FileHandle, BitmapBits, BiHeader.biSizeImage, &BytesWritten,
                  NULL);
      }
      delete[] BitmapBits;
      CloseHandle(FileHandle);
    }

    SelectObject(MemDC, OldBitmap);
    DeleteObject(MemBitmap);
    DeleteDC(MemDC);
    ReleaseDC(hwnd, ScreenDC);
  }

  return TRUE;
}

bool tryOpen(HWND hwnd) {
  OPENFILENAME OfnData;
  ZeroMemory(&OfnData, sizeof(OfnData));
  OfnData.lStructSize = sizeof(OfnData);
  OfnData.hwndOwner = hwnd;
  OfnData.lpstrFilter = L"Bitmap Files (*.bmp)\0*.bmp\0All Files (*.*)\0*.*\0";
  OfnData.nFilterIndex = 1;
  OfnData.lpstrFile = NULL;
  OfnData.nMaxFile = 0;
  OfnData.lpstrDefExt = L"bmp";
  OfnData.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

  TCHAR FileName[MAX_PATH] = L"";
  OfnData.lpstrFile = FileName;
  OfnData.nMaxFile = MAX_PATH;

  if (GetOpenFileName(&OfnData) == TRUE) {
    HDC ScreenDC = GetDC(hwnd);
    HDC MemDC = CreateCompatibleDC(ScreenDC);
    HBITMAP OldBitmapBuffer = (HBITMAP)SelectObject(hdcBuffer, hBitmapBuffer); // 현재 백 버퍼 저장
    HBITMAP BitmapHandle =
        (HBITMAP)LoadImage(NULL, FileName, IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
    if (BitmapHandle != NULL) {
      HDC FileDC = CreateCompatibleDC(ScreenDC);
      HBITMAP OldBitmap = (HBITMAP)SelectObject(FileDC, BitmapHandle);
      BitBlt(hdcBuffer, 0, 0, 1920, 1080, FileDC, 0, 0, SRCCOPY); // 새 파일의 내용을 백 버퍼에 복사

      SelectObject(FileDC, OldBitmap);
      DeleteDC(FileDC);
      DeleteObject(BitmapHandle);
    }
    SelectObject(hdcBuffer, OldBitmapBuffer); // 이전 백 버퍼로 복원
    ReleaseDC(hwnd, ScreenDC);
    InvalidateRect(hwnd, NULL, FALSE); // 화면 갱신
  }

  return TRUE;
}
