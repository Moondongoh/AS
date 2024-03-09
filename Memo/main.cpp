/*
1. 텍스트 입력 받을 수 있어야함. >> 1차원 배열 ㄴㄴ 2차원 배열로 메모리는 동적할당 받아서 해야함.
2. 메모장 상단에 메뉴바(툴바)가 있어야함
2-1. 메뉴바는 파일, 편집, 서식으로 구성 되어야함.
+ 새로 만들기, 새 창 열기, 저장, 열기 ...
2-2. 파일, 편집, 서식에는 각 카테고리에 맞는 기능이 있어야함.
3. 키보드 기능(Home, End, Delete, Insert, Page Up, Page Down, Tab, Back_Space, Enter, Up,Down,left,Right)을 인식 해야함.
4. 메모장 스크롤 기능이 있어야함. >> 윈도우 창 크기에 맞게 조절이 되어야함.
5. 캐럿이 존재해야함. 
6.     
SetScrollRange(hWnd, SB_VERT, 0, MAX_ROWS - 1, TRUE);
SetScrollRange(hWnd, SB_HORZ, 0, MAX_COLUMNS - 1, TRUE);
*** 사용한 메모리는 반납될 수 있도록 해야함 (메모리 누수 방지) ***

문제점
ㅗ Enter(줄 바꿈) 전까지 현재 입력 중인 줄의 텍스트가 안보임
개 지랄 맞은 경우 창에 글자를 작성 후 줄 바꾼 후 글자를 다 지우고 다시 작성하면 글자가 작성되면서 보임 ㅅㅂ
ㅗ 새 창 열기 후 새 창을 닫고 기존 창으로 넘어가면 새 창의 데이터가 남음
ㅗ 메세지 창 처리 안함
ㅗ 캐럿이 아직도 병50임 영어만 쓰면 맞는데 여기다 띄어쓰기랑 한글 이런거 더하면 캐럿이 문장의 끝을 못따라감.
  >> 글꼴 수정으로 영문의 폭은 찾아서 해결함 (띄어쓰기 포함) but 한글은 아직 안됌.
ㅗ 열기로 파일을 열고 해당 파일 데이터 수정이 불가 ㅅㅂ
ㅗ 스크롤 기능을 열을 한칸씩 이동하는거롤 수정해볼법함
행이 가로 열이 세로

텍스트를 입력 받는 배열을 미리 생성한다면?
최대 크기의 배열을 생성해서 이용
첫번째 줄 초기화가 안된거여서?
지금 현재 무슨 일이냐 내가 개행하는게 줄 바꿈을 배열 자체를 복사를 해서 엔터를 하는데
엔터를 치고 돌아가게 되면 되는 이유는 이미 할당이 되기 때문에 글자가 보이는 것 같음
그냥 엔터 치고 상단 줄로 올라가서 텍스트를 입력하면 보이는 것으로 확인
1
*/

/* 
구현 완료
1. 한글과 영어 입력가능
2. 메뉴바 생성과 기능들 내역 출력 가능
새로 만들기, 새 창 열기, 저장 및 열기 가능
3. 캐럿 출력 (영어는 고정폭을 갖는 글꼴을 찾아서 캐럿의 위치가 맞지만 한글은 맞지 않음)
4. HOME, END, DELETE BACK SPACE, ENTER, TAB 방향키키 먹음 INSERT와 PAGE UP, DOWN 안됌
5.스크롤바 출력 가능 (구현은 안됌)
*/


/*
스크롤바 기능 구현 완료 왼쪽 오른쪽 상단 하단누르면 움직임 스크롤바 이동도 가능
*/


#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <imm.h>
#include "head.h" // 리소스 헤더 포함

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

SCROLLINFO si; 

HINSTANCE hInst;
HWND hEdit;
WCHAR **textBuffer = NULL;         // 2차원 배열
int textRows = 0;               // x값을 나타낸다라고 생각
int textColumns = 0;            // y값을 나타낸다라고 생각
int caretRow = 0;               // 캐럿의 x
int caretColumn = 0;            // 캐럿의 y/
const int MAX_ROWS = 1000;         // 최대 행 수
const int MAX_COLUMNS = 1000;      // 최대 열 수
const int TEXT_HEIGHT = 16;
const int PAGE_SIZE = 10;         // 한번에 스크롤한 페이지
int tabWidth = 4;

int textPosX = 0;
int textPosY = 0;
POINT CaretP;

static int x,y;


// 새 창을 생성하는 함수
HWND CreateNewMemoWindow(HINSTANCE hInstance) 
{
    textRows = 0;      // 텍스트 행 수 초기화
    textColumns = 0;   // 텍스트 열 수 초기화
    caretRow = 0;      // 커서 행 위치 초기화
    caretColumn = 0;   // 커서 열 위치 초기화
    WNDCLASS wc = {0};

    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = TEXT("MyMemoClass");

    RegisterClass(&wc); //윈도우 등록 후

   // 윈도우를 생성
    HWND hWnd = CreateWindow(wc.lpszClassName, TEXT("동오의 비밀의 새 창 메모장"), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 640, 480, NULL, LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)), hInstance, NULL);
    // "중요" 새 창을 화면에 표시함.
    ShowWindow(hWnd, SW_SHOWNORMAL); 


    // 메시지 루프 >> 이벤트를 순서대로 반복문으로 처리함.
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) 
   {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return hWnd;

}

// WinMain 함수
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) 
{
    MSG msg;
    WNDCLASS wc = {0};

    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = TEXT("MyMemoClass");

    if (!RegisterClass(&wc)) return 0; //윈도우 등록

   // 윈도우 생성
    HWND hWnd = CreateWindow(
        wc.lpszClassName,                        // 윈도우 클래스 이름
        TEXT("동오의 비밀의 메모장"),               // 윈도우 제목
        WS_OVERLAPPEDWINDOW | WS_VSCROLL | WS_HSCROLL,   // 윈도우 스타일에 수직 및 수평 스크롤바 추가 기능 추가해야지 구현됌
        CW_USEDEFAULT,                           // x 위치
        CW_USEDEFAULT,                           // y 위치
        640,                                 // 너비
        480,                                 // 높이
        NULL,                                 // 부모 윈도우 핸들
        LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)),// 메뉴 핸들
        hInstance,                              // 인스턴스 핸들
        NULL                                 // 추가 매개변수
    );

    if (!hWnd) return 0;

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    // 텍스트 버퍼를 할당
    textBuffer = (WCHAR**)malloc(MAX_ROWS * sizeof(WCHAR*));
    if (textBuffer == NULL) 
   {
        MessageBox(NULL, TEXT("메모리 할당에 실패했습니다."), TEXT("에러"), MB_OK | MB_ICONERROR);
        return 0;
    }
    for (int i = 0; i < MAX_ROWS; ++i) 
   {
        textBuffer[i] = (WCHAR*)malloc(MAX_COLUMNS * sizeof(WCHAR));
        if (textBuffer[i] == NULL) 
      {
            MessageBox(NULL, TEXT("메모리 할당에 실패했습니다."), TEXT("에러"), MB_OK | MB_ICONERROR);
            return 0;
        }
        textBuffer[i][0] = L'\0'; // 각 행의 첫 번째 열에 NULL 문자 삽입
      textRows = 1; // 초기에 한 행만 할당되었음을 표시 <<< 한개를 할당한 상태
    }

    // 메시지 루프 >> 이벤트를 순서대로 반복문으로 처리함.
    while (GetMessage(&msg, NULL, 0, 0)) 
   {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // 할당된 메모리 해제
    for (int i = 0; i < MAX_ROWS; ++i) 
   {
        free(textBuffer[i]);
    }
    free(textBuffer);

    return (int)msg.wParam;
}

// 윈도우 프로시저 >> 여기가 메시지를 처리하는 곳 각 케이스로 가서 처리
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) 
{

    switch (message) 
   {
      /// 
   case WM_LBUTTONDOWN:
      if (wParam & MK_CONTROL) 
      {
         x=0;y=0;
      } else 
      {
         x=LOWORD(lParam);
         y=HIWORD(lParam);
      }
          
      SetCaretPos(x, y); // 사용자가 마우스 좌 클릭한 위치로 캐럿 이동
      ShowCaret(hWnd);

      return 0;
      ///

        case WM_CREATE:
         ///******///
            CreateCaret(hWnd, NULL, 1, TEXT_HEIGHT);
            ShowCaret(hWnd);
            SetCaretPos(0, 0);
            caretRow = 0;
            caretColumn = 0;
         SetScrollRange(hWnd, SB_VERT, 0, MAX_ROWS - 1, TRUE);
         SetScrollRange(hWnd, SB_HORZ, 0, MAX_COLUMNS - 1, TRUE);
            break;

        case WM_COMMAND:
            switch (LOWORD(wParam)) 
         {
                case ID_FILE_NEW:
                    // 텍스트 초기화
                    for (int i = 0; i < MAX_ROWS; ++i) 
               {
                        textBuffer[i][0] = L'\0';
                    }
                    textRows = 0;
                    textColumns = 0;
                    caretRow = 0;
                    caretColumn = 0;
                    SetCaretPos(0, 0);
                    InvalidateRect(hWnd, NULL, TRUE);
                    break;

                case ID_FILE_NEW_WINDOW:
                    // 새 창 열기
                    CreateNewMemoWindow(hInst); 
                    break;

            
            case ID_FILE_SAVE:
               
            {
                  OPENFILENAMEW save; // 유니코드 버전의 OPENFILENAME 구조체 사용
                  
                  WCHAR szFileName[MAX_PATH] = L"";

                  ZeroMemory(&save, sizeof(save));      //변수를 초기화하기 위해 사용
                  save.lStructSize = sizeof(save);
                  save.hwndOwner = hWnd;
                  save.lpstrFilter = L"텍스트 파일 (*.txt)\0*.TXT\0모든 파일 (*.*)\0*.*\0";
                  save.lpstrFile = szFileName;         // 파일 이름을 저장할 버퍼
                  save.lpstrFile[0] = L'\0';         // 파일 이름 버퍼 초기화
                  save.nMaxFile = sizeof(szFileName) / sizeof(*szFileName);
                  save.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;

                  if (GetSaveFileNameW(&save) == TRUE) 

              { // GetSaveFileNameW 함수 사용하여 유니코드로 파일 이름 처리
                     FILE* file = _wfopen(save.lpstrFile, L"w, ccs=UTF-8"); // 파일 이름을 유니코드로 처리하여 열기
                     if (file != NULL) 
                {
                        // 모든 행에 대해 파일에 쓰기
                        for (int i = 0; i < textRows; ++i) 
                  {
                           fwprintf(file, L"%s\n", textBuffer[i]); // 유니코드로 파일에 쓰기
                  }
                        fclose(file); // 파일 닫기
                }
              }
            }
               break;

            case ID_FILE_OPEN:
               {
    
                  OPENFILENAMEW open; // 유니코드 버전의 OPENFILENAME 구조체 사용

                  WCHAR szFileName[MAX_PATH] = L"";

                  ZeroMemory(&open, sizeof(open)); //저장과 동일하게 초기화
    
                  open.lStructSize = sizeof(open);
                  open.hwndOwner = hWnd;
                  
                  open.lpstrFilter = L"텍스트 파일 (*.txt)\0*.TXT\0모든 파일 (*.*)\0*.*\0";
                  open.lpstrFile = szFileName;
                  open.lpstrFile[0] = L'\0';
                  open.nMaxFile = sizeof(szFileName) / sizeof(*szFileName);
                  open.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
                  
                  if (GetOpenFileNameW(&open) == TRUE) 
              { // GetOpenFileNameW 함수 사용하여 유니코드로 파일 이름 처리
                     FILE* file = _wfopen(open.lpstrFile, L"r, ccs=UTF-8"); // 파일 이름을 유니코드로 처리하여 열기
                     if (file != NULL) 
                {
                        // 텍스트를 저장할 임시 버퍼 할당
                        WCHAR tempBuffer[MAX_COLUMNS];
                        while (fgetws(tempBuffer, MAX_COLUMNS, file) != NULL)  // 한 줄 한 줄 읽어옴 NULL까지
                  
                  {
                           tempBuffer[wcslen(tempBuffer) - 1] = L'\0'; // 개행 문자 제거
                           wcscpy_s(textBuffer[textRows], MAX_COLUMNS, tempBuffer); //읽은 거 복사하기 줄 수 올리면서
                           textRows++;
                        }
                        fclose(file); // 파일 닫기

                        // 캐럿 생성
                        CreateCaret(hWnd, NULL, 1, TEXT_HEIGHT);
                        ShowCaret(hWnd);
                        SetCaretPos(0, 0);
                        caretRow = 0;
                        caretColumn = 0;
                        InvalidateRect(hWnd, NULL, TRUE); // 윈도우를 다시 그리도록 강제
                     
                     }
                  }
               }
               break;
                
         case ID_FILE_EXIT:
            DestroyWindow(hWnd);
            break;
      }         
      break;
     ///
     case WM_HSCROLL:
        ZeroMemory(&si, sizeof(si));
        si.cbSize = sizeof(si);
        si.fMask = SIF_ALL;
        GetScrollInfo(hWnd, SB_HORZ, &si);

        switch (LOWORD (wParam))
        
        {
        // 왼쪽 버튼 누르면 움직임 하지만 왼쪽 한계점 설정
        case SB_LINELEFT:
         if(textPosX == 0)
         {
         break;
         }
         GetCaretPos(&CaretP);
            textPosX -= 8;
         SetCaretPos(CaretP.x+8, CaretP.y);
         si.nPos -= 8;
         SetScrollInfo(hWnd, SB_HORZ, &si, TRUE); // Add this line
         InvalidateRect(hWnd, NULL, TRUE);
         UpdateWindow(hWnd);
            break;
              
        // 오른쪽 버튼 누르면 움직임.
        case SB_LINERIGHT:
         if(MAX_ROWS == 0)
         {
         break;
         }
         
         GetCaretPos(&CaretP);
         textPosX += 8;
         SetCaretPos(CaretP.x-8, CaretP.y);
         si.nPos += 8;
         SetScrollInfo(hWnd, SB_HORZ, &si, TRUE); // Add this line
         InvalidateRect(hWnd, NULL, TRUE);
         UpdateWindow(hWnd);
       break;
      
      case SB_THUMBTRACK:
         si.nPos = si.nTrackPos;
         textPosX = si.nPos;
         SetScrollInfo(hWnd, SB_HORZ, &si, TRUE);
         InvalidateRect(hWnd, NULL, TRUE);
         UpdateWindow(hWnd);
         break;
        }
        break;
        
     case WM_VSCROLL:
        
        ZeroMemory(&si, sizeof(si));
        si.cbSize = sizeof(si);
        si.fMask = SIF_ALL;
        GetScrollInfo(hWnd, SB_VERT, &si);

        
   switch (LOWORD (wParam))
      
   {
        // 위 버튼 누르면 움직임.
      case SB_LINEUP:
         if(textPosY == 0)
       {
         break;
         }
         GetCaretPos(&CaretP);
         textPosY -= 16;
         si.nPos -= 16; // Add this line
         SetScrollInfo(hWnd, SB_VERT, &si, TRUE);
         SetCaretPos(CaretP.x, CaretP.y+16);
         InvalidateRect(hWnd, NULL, TRUE);
         UpdateWindow(hWnd);
         break;
              
        // 아래 버튼 누르면 움직임.
        case SB_LINEDOWN:
         
         GetCaretPos(&CaretP);
         textPosY += 16;
         si.nPos += 16; // Add this line
         SetScrollInfo(hWnd, SB_VERT, &si, TRUE);
         SetCaretPos(CaretP.x, CaretP.y-16);
         InvalidateRect(hWnd, NULL, TRUE);
         UpdateWindow(hWnd);
         break;

      
      
      case SB_THUMBTRACK:
         
         si.nPos = si.nTrackPos;
         textPosY = si.nPos;
         SetScrollInfo(hWnd, SB_VERT, &si, TRUE);
         InvalidateRect(hWnd, NULL, TRUE);
         UpdateWindow(hWnd);
         break;
   }
        
   break;

      case WM_SIZE:
         {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            SetTextColor(hdc, RGB(0, 0, 0));
            SetBkColor(hdc, RGB(255, 255, 255));

         HFONT hFont = CreateFont(17, 0, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_OUTLINE_PRECIS,
                          CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FIXED_PITCH | FF_MODERN, TEXT("Courier New"));
         HFONT hOldFont = (HFONT)SelectObject(hdc, hFont);
         //FF_MODERN는 고정폭 글꼴을 의미하는 플래그이며, Courier New는 고정폭 글꼴의 한 코드임.
         //문제점 Courier New << 이친구는 영문만 고정폭 지원

            // 모든 텍스트를 출력
            for (int i = 0; i < textRows; ++i) 
         {
                TextOut(hdc, 0, TEXT_HEIGHT * i, textBuffer[i], wcslen(textBuffer[i]));
         }
         SelectObject(hdc, hOldFont);
         DeleteObject(hFont);
            EndPaint(hWnd, &ps);
            break;
        }
            break;

        case WM_KEYDOWN:
            switch (wParam) 
         {
                case VK_HOME:
                    caretColumn = 0;
                    break;

                case VK_END:
                    caretColumn = textColumns;
                    break;

                case VK_LEFT:
                    if (caretColumn > 0) caretColumn--;
                    break;

                case VK_RIGHT:
                    if (caretColumn < textColumns) caretColumn++;
                    break;

                case VK_UP:
                    if (caretRow > 0) caretRow--;
                    break;

                case VK_DOWN:
                    if (caretRow < textRows) caretRow++;
                    break;
                    
            case VK_PRIOR: // 페이지 업
               // 캐럿 행을 표시된 행의 수만큼 위로 스크롤.
               caretRow -= textRows;
               // 행 값이 0보다 작으면 최상단이 0까지가도록 하기위해서 넣음.
               if (caretRow < 0) caretRow = 0;
               break;
        
            case VK_NEXT: // 페이지 다운
               caretRow += textRows-1;
               // 현재 캐럿이 텍스트 캐럿보다 밑에 형성될때 캐럿은 현재 존재하는 행의 좌표.
               if (caretRow > textRows) caretRow = textRows;
               break;

                case VK_TAB:
                    // 탭 문자 대신 스페이스를 넣어 일정 간격을 벌려줍니다.
                    for (int i = textColumns; i > caretColumn; i--) 
               {
                        textBuffer[caretRow][i + tabWidth] = textBuffer[caretRow][i];
                    }
                    // 일정 간격만큼 스페이스를 넣어줍니다.
                    for (int i = 0; i < tabWidth; ++i) 
               {
                        textBuffer[caretRow][caretColumn + i] = L' ';
                    }
                    caretColumn += tabWidth;
                    textColumns += tabWidth;
                    break;

                
            case VK_RETURN:
    
               // 엔터를 누르면 특정 동작 수행
               // 예를 들어, 새로운 줄로 이동하고 새로운 줄을 미리 채워넣을 수 있습니다.
               // 이 코드에서는 새로운 줄로 이동하고 빈 줄을 추가합니다.
               // 새로운 행 추가
               for (int i = textRows; i > caretRow; i--) 
            {
                  wcscpy_s(textBuffer[i], MAX_COLUMNS, textBuffer[i - 1]);
               }
               textRows++;
               caretRow++;

               // 새로운 행 초기화
               for (int j = MAX_COLUMNS - 1; j > 0; j--) 
            {
                  textBuffer[caretRow][j] = textBuffer[caretRow - 1][j - 1];
               }
               textBuffer[caretRow][0] = L'\0'; // 새로운 줄의 첫 번째 열에 NULL 문자 삽입
               caretColumn = 0;
               textColumns = 0; // 새로운 줄이므로 열 수를 0으로 설정합니다. << 문자열을 복사해서 다음줄에 배정하기 때문에 복사한 문자열의 문자를 모두 지워주어야해서 텍스트칼럼을 비운다.
               break;


            case VK_BACK:
               // 백스페이스를 누르면 커서의 왼쪽에 있는 문자를 삭제합니다.
               if (caretColumn > 0) 
            {
                  for (int i = caretColumn - 1; i < textColumns; i++) 
              {
                     textBuffer[caretRow][i] = textBuffer[caretRow][i + 1];
                  }
                  caretColumn--;
                  textColumns--;
               } else if (caretRow > 0) 
            {  // 줄 바꿈을 고려합니다.
                  caretRow--; //Row를 하나 줄이고 그 줄 끝에서 왼쪽으로 하나씩 삭제하도록
                  caretColumn = wcslen(textBuffer[caretRow]);  // 이전 줄의 길이로 caretColumn을 설정합니다.
                  textColumns = caretColumn;
               }
               break;

                case VK_DELETE:
                    // 딜리트를 누르면 커서의 위치에 있는 문자를 삭제합니다.
                    if (caretColumn < textColumns) 
               {
                        for (int i = caretColumn; i < textColumns; i++) 
                  {
                            textBuffer[caretRow][i] = textBuffer[caretRow][i + 1];
                        }
                        textColumns--;
                    }
                    break;
            }

            //SetCaretPos(8 * caretColumn, TEXT_HEIGHT * caretRow); //8은 폰트 너비 이걸 조절?
            if  ((wParam >= 0xAC00 && wParam <= 0xD7AF))
            SetCaretPos(15 * caretColumn, TEXT_HEIGHT * caretRow); // 한글 문자의 경우
         else
            SetCaretPos(8 * caretColumn, TEXT_HEIGHT * caretRow); // 이외의 문자의 경우
         InvalidateRect(hWnd, NULL, TRUE);
         break;

      case WM_CHAR:
         
         if (textRows < MAX_ROWS && textColumns < MAX_COLUMNS) 
       {
            if ((wParam >= 0x20 && wParam <= 0x7E) ||   // ASCII 문자 범위 이게 영어
               (wParam >= 0xAC00 && wParam <= 0xD7AF)) 
         {  // 한글 유니코드 범위
                  
                  // 버퍼 오버플로우 방지
                  if (textColumns < MAX_COLUMNS - 1) 
              {
                     
                     // 삽입할 문자 이후의 문자들을 오른쪽으로 이동
                     for (int i = textColumns; i > caretColumn; i--) 
                {
                        textBuffer[caretRow][i] = textBuffer[caretRow][i - 1];
                     }
                
                     // 입력된 문자를 삽입
                     textBuffer[caretRow][caretColumn] = (WCHAR)wParam;
                     caretColumn++;
                     textColumns++;
                  }
            }
         }
    
         // NULL 문자를 추가 << 이 줄 넣으니 첫 줄에 이상한 단어 들어가는 현상 사라짐
         // TEXTOut() 함수는 출력할 문자와 그 문자열으 길이를 인자로 받음 하지만 이때 문자열의 길이를 정학하게 알아야함.
         // 그래서 NULL을 통해 끝을 맺어줘야 길이를 알 수 있기에 NULL값 을 넣어주는 코드로 해결
         
         textBuffer[caretRow][textColumns] = '\0';

         if  ((wParam >= 0xAC00 && wParam <= 0xD7AF))
            SetCaretPos(15 * caretColumn, TEXT_HEIGHT * caretRow); // 한글 문자의 경우
         
         else
            SetCaretPos(8 * caretColumn, TEXT_HEIGHT * caretRow); // 이외의 문자의 경우
         InvalidateRect(hWnd, NULL, TRUE);
         //UpdateWindow(hWnd);
         break;



        case WM_PAINT: 
      {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            SetTextColor(hdc, RGB(0, 0, 0));
            SetBkColor(hdc, RGB(255, 255, 255));

         HFONT hFont = CreateFont(17, 0, 0, 0, FW_DONTCARE, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_OUTLINE_PRECIS,
                          CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FIXED_PITCH | FF_MODERN, TEXT("Courier New"));
         HFONT hOldFont = (HFONT)SelectObject(hdc, hFont);
         //FF_MODERN는 고정폭 글꼴을 의미하는 플래그이며, Courier New는 고정폭 글꼴의 한 코드임.
         //문제점 Courier New << 이친구는 영문만 고정폭 지원

            // 모든 텍스트를 출력
            for (int i = 0; i < textRows; ++i) 
         {
                TextOut(hdc, 0 - textPosX, TEXT_HEIGHT * i - textPosY, textBuffer[i], wcslen(textBuffer[i]));
            }
         SelectObject(hdc, hOldFont);
         DeleteObject(hFont);

            EndPaint(hWnd, &ps);
            break;
        
      }

      case WM_KILLFOCUS:

         HideCaret(hWnd);
         DestroyCaret();
              break;
   
      case WM_SETFOCUS:
      {
         CreateCaret(hWnd,NULL,1, 17);
         ShowCaret(hWnd);
      }
      break;

        case WM_DESTROY:
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;

}
